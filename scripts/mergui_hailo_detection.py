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
    print(">>> THREAD: Type a number (ID) to track, or -1 to see all.\n")

    while True:
        try:
            # This line waits for you to type in the terminal
            user_input = input()  # Waits for Enter

            # Convert string to integer
            new_id = int(user_input)

            # Update the shared object
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

        # State Variables
        self.target_id = -1
        self.frame_counter = 0

        # Current Motor Positions (Software State)
        self.current_pan = self.center_pan
        self.current_tilt = self.center_tilt

        # --- FIX: This was missing and caused the crash ---
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


        # ======================================================
        # NEW: Non-blocking cooldown for PTZ moves
        # ======================================================
        self.last_move_time = 0.0      # time of last PTZ move
        self.move_cooldown = 0.5       # seconds between moves (tune this)
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
        print(f"# COORDINATE SYSTEM MAP:")
        print(f"# ----------------------")
        print(f"# Top-Left:     (0, 0)          -> Norm: (0.0, 0.0)")
        print(f"# Top-Right:    ({w}, 0)       -> Norm: (1.0, 0.0)")
        print(f"# Center:       ({w // 2}, {h // 2})        -> Norm: (0.5, 0.5)")
        print(f"# Bottom-Left:  (0, {h})        -> Norm: (0.0, 1.0)")
        print(f"# Bottom-Right: ({w}, {h})     -> Norm: (1.0, 1.0)")
        print("#" * 60 + "\n")
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

            # Get Coordinates
            bbox = det.get_bbox()
            center_x = bbox.xmin() + (bbox.width() / 2)
            center_y = bbox.ymin() + (bbox.height() / 2)

            print(f"*** TARGET  [{track_id}] ***: Pos: X={center_x:.2f}")

            # =========================================================
            # TRACKING LOGIC (Horizontal Only)
            # =========================================================
            if user_data.target_id != -1:

                # --- ADDED: CHECK COOLDOWN ---

                # 1. Calculate Error (Center is 0.5)
                # If X > 0.5 (Right side), Error is Positive
                error_x = center_x - 0.5

                norm_x = bbox.xmin()
                norm_y = bbox.ymin()

                pixel_x = int(norm_x * w)
                pixel_y = int(norm_y * h)

                print(f"Normalized: ({norm_x:.2f}, {norm_y:.2f})  -->  Pixels: ({pixel_x}, {pixel_y})")

                back_to_norm_x = pixel_x / w
                back_to_norm_y = pixel_y / h

                print(f"Check Back: ({back_to_norm_x:.2f}, {back_to_norm_y:.2f})")

                print("distance between object and center:", abs(error_x))
                normalized_error_x = int(center_x * w)
                print("normalized error x in pixels:", normalized_error_x)

                # 2. Deadzone (Don't move if error is small, e.g. < 5%)
                if abs(error_x) > 0.2:

                    # 3. Calculate New Pan Position
                    A_current = user_data.current_pan

                    # --- YOUR LOGIC KEPT EXACTLY AS IS ---
                    A_c = ((pixel_x - 640) / 13.3)

                    print("A_current:", A_current)
                    print("A_c:", A_c)
                    new_pan = int(A_current + A_c)
                    print("new_pan:", new_pan)
                    # 4. Clamp Limits (0 to 1000)
                    new_pan = max(0, min(180, new_pan))

                    # 5. Move Motor (Only if changed significantly)
                    if abs(new_pan - user_data.current_pan) > 2:
                        # --------------------------------------------------------
                        # Non-blocking cooldown using time.monotonic()
                        # --------------------------------------------------------
                        now = time.monotonic()
                        if now - user_data.last_move_time >= user_data.move_cooldown:
                            print("[TRACK] Moving to new pan...")
                            user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
                            user_data.current_pan = new_pan
                            user_data.last_move_time = now
                        else:
                            # Optional: debug to see how often it wants to move
                            remaining = user_data.move_cooldown - (now - user_data.last_move_time)
                            print(f"[TRACK] Cooldown active, skipping move ({remaining:.2f}s left)")

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

    # Input Thread
    input_t = threading.Thread(target=user_input_loop, args=(user_data,), daemon=True)
    input_t.start()

    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass