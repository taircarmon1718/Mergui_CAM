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
# 1. INPUT THREAD (Runs in parallel)
# =====================================================================
def user_input_loop(user_data):
    """
    This runs in the background. It waits for you to type an ID.
    """
    print("\n>>> THREAD: Input listener started.")
    print(">>> THREAD: Type a number (ID) to track, or -1 to stop tracking.\n")

    while True:
        try:
            # This line waits for you to type in the terminal
            print("Enter ID to track > ", end="", flush=True)
            user_input = input()  # Waits for Enter

            # Convert string to integer
            new_id = int(user_input)

            # Update the shared object
            user_data.target_id = new_id

            if new_id == -1:
                print(f" [SYSTEM] Tracking Mode: IDLE (Monitoring only)")
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

        # State Variables
        self.target_id = -1
        self.frame_counter = 0

        # --- NEW: Cooldown counter for tracking ---
        self.track_wait_counter = 0

        # Current Motor Positions (Software State)
        self.current_pan = self.center_pan
        self.current_tilt = self.center_tilt

        self.track_gain = 30  # Controls tracking speed

        # Autofocus logic
        self.is_focusing = True
        self.focuser.set(Focuser.OPT_FOCUS, 200)

        print("[INIT] Starting Initial AutoFocus...")

        # Using the Library AutoFocus as you requested
        self.autofocus = AutoFocus(self.focuser, camera=None)
        self.autofocus.debug = True
        self.autofocus.startFocus_hailo()

        self.info_printed = False
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

    # --- DECREMENT TRACKING COOLDOWN ---
    if user_data.track_wait_counter > 0:
        user_data.track_wait_counter -= 1

    fmt, w, h = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    # -------------------------------------------------------------
    # PRINT SCREEN INFO ONCE (When first frame arrives)
    # -------------------------------------------------------------
    if not user_data.info_printed:
        print("\n" + "#" * 60)
        print(f"# VIDEO STREAM STARTED SUCCESSFULLY")
        print(f"# RESOLUTION: {w} x {h} pixels")
        print("#" * 60)
        user_data.info_printed = True

    if frame is None:
        return Gst.PadProbeReturn.OK

    # --- AUTOFOCUS STEP (Existing Logic) ---
    if user_data.is_focusing:
        finished, best_pos = user_data.autofocus.stepFocus_hailo(frame)
        if finished:
            print(f"!!! [AF-H] FINISHED! Best Focus: {best_pos} !!!")
            user_data.is_focusing = False
            user_data.focuser.set(Focuser.OPT_FOCUS, best_pos)

    # --- DETECTION & TRACKING ---
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for det in detections:
        label = det.get_label()
        confidence = det.get_confidence()

        if label == "person" and confidence > 0.5:
            # Get ID
            track_id = -1
            for obj in det.get_objects_typed(hailo.HAILO_UNIQUE_ID):
                track_id = obj.get_id()

            # Filter by Target ID
            if user_data.target_id != -1:
                if track_id != user_data.target_id:
                    continue  # Skip detection if it's not the target

                # If we are here, we found the target!

                # Get Coordinates
                bbox = det.get_bbox()
                center_x = bbox.xmin() + (bbox.width() / 2)

                print(f"*** TRACKING ID [{track_id}] ***: Pos: X={center_x:.2f}")

                # =========================================================
                # TRACKING LOGIC (Horizontal Only)
                # =========================================================

                # --- CHECK COOLDOWN: If we moved recently, WAIT. ---
                if user_data.track_wait_counter > 0:
                    continue  # Skip motor updates this frame

                # 1. Calculate Error (Center is 0.5)
                # If X > 0.5 (Right side), Error is Positive
                error_x = center_x - 0.5

                norm_x = bbox.xmin()
                pixel_x = int(norm_x * w)

                # 2. Deadzone (Don't move if error is small, e.g. < 20%)
                if abs(error_x) > 0.2:

                    # 3. Calculate New Pan Position using your formula
                    A_current = user_data.current_pan

                    # Your logic: (pixel_x - 640) / 13.3
                    # This calculates the EXACT angle difference
                    A_c = ((pixel_x - 640) / 13.3) /2

                    print(f"Current Pan: {A_current} | Correction (Ac): {A_c:.2f}")

                    new_pan = int(A_current + A_c)
                    print("Calculated New Pan:", new_pan)

                    # 4. Clamp Limits (0 to 180 degrees)
                    new_pan = max(0, min(180, new_pan))

                    # 5. Move Motor (Only if changed significantly to avoid jitter)
                    if abs(new_pan - user_data.current_pan) > 2:
                        user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
                        user_data.current_pan = new_pan
                        print(f">>> MOVING PAN TO: {new_pan}")

                        # --- SET COOLDOWN ---
                        # Wait 5 frames (approx 0.15s) before tracking again
                        # This solves the "over-correction" problem
                        user_data.track_wait_counter = 5

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

    # Input Thread (Daemon ensures it closes when main app closes)
    input_t = threading.Thread(target=user_input_loop, args=(user_data,), daemon=True)
    input_t.start()

    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass