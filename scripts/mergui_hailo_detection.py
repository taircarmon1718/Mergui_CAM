#!/usr/bin/env python3
import os
import sys
import time
import threading
import argparse
import gi
import cv2
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

# --- HAILO IMPORTS ---
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    app_callback_class,
)
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import (
    GStreamerDetectionApp,
)

# --- PTZ DRIVER IMPORT ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from B016712MP.Focuser import Focuser
from B016712MP.AutoFocus import AutoFocus


# =====================================================================
# 1. INPUT THREAD
# =====================================================================
def user_input_loop(user_data):
    print("\n>>> THREAD: Type a number (ID) to track, or -1 to see all.\n")
    while True:
        try:
            user_input = input()
            new_id = int(user_input)
            user_data.target_id = new_id
            if new_id == -1:
                print(f" [SYSTEM] Tracking Mode: IDLE (Show All)")
            else:
                print(f" [SYSTEM] Tracking Mode: LOCKED ON ID {new_id}")
        except ValueError:
            print(" [ERROR] Please enter a valid number.")
        except EOFError:
            break


# =====================================================================
# USER APP CLASS
# =====================================================================
class UserApp(app_callback_class):
    def __init__(self):
        super().__init__()

        print("\n" + "=" * 40)
        print("[INIT] Initializing PTZ Camera...")

        try:
            self.focuser = Focuser(1)
            self.focuser.set(Focuser.OPT_MODE, 1)
            time.sleep(0.2)
            self.focuser.set(Focuser.OPT_IRCUT, 0)
            time.sleep(0.5)

            # Start Center
            self.center_pan = 90
            self.center_tilt = 25

            print("[INIT] Setting Home Position...")
            self.focuser.set(Focuser.OPT_MOTOR_X, self.center_pan)
            self.focuser.set(Focuser.OPT_MOTOR_Y, self.center_tilt)
            time.sleep(1.0)

        except Exception as e:
            print(f"[ERROR] PTZ Init failed: {e}")

        self.target_id = -1
        self.frame_counter = 0
        self.info_printed = False

        # Software state
        self.current_pan = self.center_pan
        self.current_tilt = self.center_tilt

        # Tracking Logic
        self.pixels_per_degree = 13.3
        self.pan_direction = 1

        # --- NEW: COOLDOWN TIMER ---
        self.move_cooldown = 0  # Counter to wait between moves

        # Autofocus
        self.is_focusing = True
        self.focuser.set(Focuser.OPT_FOCUS, 200)

        print("[INIT] Starting Initial AutoFocus...")
        self.autofocus = AutoFocus(self.focuser, camera=None)
        self.autofocus.debug = True
        self.autofocus.startFocus_hailo()

        print("[INIT] Ready.")
        print("=" * 40 + "\n")


# =====================================================================
# CALLBACK FUNCTION
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    user_data.frame_counter += 1

    # Decrease cooldown
    if user_data.move_cooldown > 0:
        user_data.move_cooldown -= 1

    fmt, w, h = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    if not user_data.info_printed:
        print(f"\n# VIDEO RESOLUTION: {w} x {h}")
        print(f"# SCREEN CENTER X: {w // 2}")
        user_data.info_printed = True

    if frame is None:
        return Gst.PadProbeReturn.OK

    # --- AUTOFOCUS ---
    if user_data.is_focusing:
        finished, best_pos = user_data.autofocus.stepFocus_hailo(frame)
        if finished:
            print(f"!!! [AF] FINISHED! Best Focus: {best_pos} !!!")
            user_data.is_focusing = False
            user_data.focuser.set(Focuser.OPT_FOCUS, best_pos)

    # --- TRACKING ---
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for det in detections:
        label = det.get_label()
        confidence = det.get_confidence()

        if label == "person" and confidence > 0.5:
            track_id = -1
            for obj in det.get_objects_typed(hailo.HAILO_UNIQUE_ID):
                track_id = obj.get_id()

            if user_data.target_id != -1 and track_id != user_data.target_id:
                continue

                # Calculate Center
            bbox = det.get_bbox()
            norm_center_x = bbox.xmin() + (bbox.width() / 2)

            # 1. Convert to Pixels
            pixel_x = int(norm_center_x * w)

            # 2. Calculate Error
            screen_center_x = w // 2
            pixel_error = pixel_x - screen_center_x

            # Print status
            if user_data.frame_counter % 10 == 0:  # Print only sometimes to reduce spam
                print(f"*** ID [{track_id}] *** X: {pixel_x} | Err: {pixel_error}px")

            # --- MOVE LOGIC (With Cooldown) ---
            if user_data.move_cooldown == 0:

                # 3. Calculate Degrees
                degrees_to_move = pixel_error / user_data.pixels_per_degree
                adjustment = degrees_to_move * user_data.pan_direction

                # Only move if error is significant (> 2 degrees)
                if abs(adjustment) > 2:
                    new_pan = int(user_data.current_pan + adjustment)

                    # Clamp
                    new_pan = max(0, min(180, new_pan))

                    if new_pan != user_data.current_pan:
                        print(f"   >>> MOVING: {adjustment:.1f}deg -> NewPan: {new_pan}")

                        user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
                        user_data.current_pan = new_pan

                        # SET COOLDOWN: Wait 15 frames (0.5 sec) before moving again!
                        user_data.move_cooldown = 15

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rpi", help="Input source")
    args, unknown = parser.parse_known_args()
    args.input = "rpi"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")
    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    print("[MAIN] Starting Pipeline...")

    user_data = UserApp()
    input_t = threading.Thread(target=user_input_loop, args=(user_data,), daemon=True)
    input_t.start()

    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass