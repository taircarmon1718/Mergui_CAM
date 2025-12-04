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
                print(f" [SYSTEM] Tracking Mode: ALL TARGETS")
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
    """
    Holds all user-side state:
      - PTZ focuser
      - frame counter
      - autofocus object and flags
      - target_id
    """

    def __init__(self):
        super().__init__()

        print("\n" + "=" * 40)
        print("[INIT] Initializing PTZ Camera...")

        try:
            # Create focuser driver
            self.focuser = Focuser(1)

            # Set PTZ mode and open IR-cut filter
            self.focuser.set(Focuser.OPT_MODE, 1)
            time.sleep(0.2)
            self.focuser.set(Focuser.OPT_IRCUT, 0)
            time.sleep(0.5)

            # Start at a known pan/tilt "home" position
            print("[INIT] Setting Home Position (Pan: 0, Tilt: 25)...")
            self.focuser.set(Focuser.OPT_MOTOR_X, 0)
            self.focuser.set(Focuser.OPT_MOTOR_Y, 25)
            time.sleep(1.0)

        except Exception as e:
            print(f"[ERROR] PTZ Init failed: {e}")

        # --- ID SELECTION VARIABLE ---
        self.target_id = -1  # -1 means show everything

        # Frame counter for timing-based logic
        self.frame_counter = 0
        self.has_moved_once = False

        # Autofocus logic
        self.is_focusing = True
        self.focuser.set(Focuser.OPT_FOCUS, 200)  # Initial Guess

        print("[INIT] Starting Initial AutoFocus...")

        self.autofocus = AutoFocus(self.focuser, camera=None)
        self.autofocus.debug = True
        self.autofocus.startFocus_hailo()

        print("[INIT] Ready.")
        print("=" * 40 + "\n")


# =====================================================================
# CALLBACK FUNCTION (Runs ~30 times per second)
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    """
    This function is called for each buffer that passes through the pipeline.
    """
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Read ROI metadata (Detections + IDs)
    roi = hailo.get_roi_from_buffer(buffer)

    # Increase frame counter
    user_data.frame_counter += 1

    # Extract the current video frame as a NumPy array
    fmt, w, h = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer, fmt, w, h)
    if frame is None:
        return Gst.PadProbeReturn.OK

    # -----------------------------------------------------------
    # INCREMENTAL AUTOFOCUS (non-blocking)
    # -----------------------------------------------------------
    if user_data.is_focusing:
        finished, best_pos = user_data.autofocus.stepFocus_hailo(frame)

        if finished:
            print(f"!!! [AF-H] FINISHED! Best Focus Position: {best_pos} !!!")
            user_data.is_focusing = False
            user_data.focuser.set(Focuser.OPT_FOCUS, best_pos)

    # -----------------------------------------------------------
    # DETECTION & ID LOGIC
    # -----------------------------------------------------------
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for det in detections:
        label = det.get_label()
        confidence = det.get_confidence()

        if label == "person" and confidence > 0.5:
            # --- GET ID ---
            track_id = -1
            # Extract Unique ID from the detection object
            for obj in det.get_objects_typed(hailo.HAILO_UNIQUE_ID):
                track_id = obj.get_id()

            # --- FILTER BY ID ---
            # If we are looking for a specific ID (not -1)
            if user_data.target_id != -1:
                if track_id != user_data.target_id:
                    continue  # Skip this person if it's not the one we want

            # If we reached here, it's either the target ID or we are showing everyone

            bbox = det.get_bbox()
            center_x = bbox.xmin() + (bbox.width() / 2)
            center_y = bbox.ymin() + (bbox.height() / 2)

            prefix = ">>> DETECTED"
            if user_data.target_id != -1:
                prefix = f"*** TARGET LOCKED [{track_id}] ***"

            print(f"{prefix}: Person | ID: {track_id} | Pos: X={center_x:.2f}, Y={center_y:.2f}")

    # Always return OK so GStreamer continues
    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rpi", help="Input source")
    args, unknown = parser.parse_known_args()
    args.input = "rpi"

    # Configure environment for Hailo pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")
    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    print("[MAIN] Starting Pipeline...")

    user_data = UserApp()

    # Start the input thread (Pass user_data so it can update target_id)
    input_t = threading.Thread(target=user_input_loop, args=(user_data,), daemon=True)
    input_t.start()

    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass