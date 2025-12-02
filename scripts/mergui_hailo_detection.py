#!/usr/bin/env python3
import os
import sys
import time
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

            # 1. Start at CENTER (0)
            print("[INIT] Setting Home Position (Pan: 0, Tilt: 25)...")
            self.focuser.set(Focuser.OPT_MOTOR_X, 0)
            self.focuser.set(Focuser.OPT_MOTOR_Y, 25)
            time.sleep(1.0)

        except Exception as e:
            print(f"[ERROR] PTZ Init failed: {e}")

        # Flags for our logic
        self.frame_counter = 0
        self.has_moved_once = False  # Flag to ensure we only move once

        # Auto-Focus variables
        self.af_running = True
        self.af_pos = 0
        self.af_step = 20
        self.af_max = 600
        self.af_best_val = 0.0
        self.af_best_pos = 0
        self.af_skip_counter = 0
        self.af_finished = False

        print("[INIT] Ready. Camera will move RIGHT in ~2 seconds.")
        print("=" * 40 + "\n")


# =====================================================================
# CALLBACK FUNCTION (Runs ~30 times per second)
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increment frame counter
    user_data.frame_counter += 1

    # -----------------------------------------------------------
    # AUTOMATIC MOVEMENT LOGIC (Time-Based)
    # -----------------------------------------------------------
    # 60 frames is approximately 2 seconds (at 30 FPS)
    if user_data.frame_counter == 90 and not user_data.has_moved_once:
        print("\n>>> [TIMER] 3 Seconds passed! Moving Camera to Right...")

        # Perform the move (Pan to 300)
        user_data.focuser.set(Focuser.OPT_MOTOR_X, 90)

        print(">>> [TIMER] Move to 90 Complete.")
        print(">>> [TIMER] Camera will move BACK to Center in 2 seconds...")
        time.sleep(2.0)
        user_data.focuser.set(Focuser.OPT_MOTOR_X, 180)
        print(">>> [TIMER] Moving Camera back to 180...")
        time.sleep(2.0)
        print(">>> [TIMER] Move to Center Complete.")
        user_data.focuser.set(Focuser.OPT_MOTOR_X, 0)
        print(">>> [TIMER] Camera will move BACK to Center in 2 seconds...")
        # Mark as done so we don't do it again
        user_data.has_moved_once = True
        print(">>> [TIMER] Move Complete.\n")

    # -----------------------------------------------------------
    # AUTO-FOCUS LOGIC
    # -----------------------------------------------------------
    if user_data.af_running and not user_data.af_finished:
        if user_data.frame_counter % 3 == 0:
            fmt, w, h = get_caps_from_pad(pad)

            # Read-Only access (Safe)
            frame = get_numpy_from_buffer(buffer, fmt, w, h)

            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()

                if score > user_data.af_best_val:
                    user_data.af_best_val = score
                    user_data.af_best_pos = user_data.af_pos

                user_data.af_pos += user_data.af_step

                if user_data.af_pos <= user_data.af_max:
                    user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_pos)
                else:
                    print(f"[AF] DONE. Best Focus: {user_data.af_best_pos}")
                    user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                    user_data.af_finished = True
                    user_data.af_running = False

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

    print(f"[MAIN] Starting Pipeline...")

    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass