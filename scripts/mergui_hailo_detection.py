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
from B016712MP.AutoFocus import AutoFocus



# =====================================================================
# USER APP CLASS
# =====================================================================
class UserApp(app_callback_class):
    """
    Holds all user-side state:
      - PTZ focuser
      - frame counter
      - autofocus object and flags
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

        # Frame counter for timing-based logic (e.g., 3-second timer)
        self.frame_counter = 0
        self.has_moved_once = False  # ensure we only do the 3s move once

        # Old autofocus flags (kept so nothing external breaks)
        self.af_running = True
        self.af_pos = 0
        self.af_step = 20
        self.af_max = 600
        self.af_best_val = 0.0
        self.af_best_pos = 0
        self.af_skip_counter = 0
        self.af_finished = False
        self.af_enabled = True
        self.af_direction = +15
        self.af_current_pos = 200
        self.af_last_score = -1
        self.af_done = False

        # Autofocus is active at startup
        self.is_focusing = True

        # Initial focus guess
        self.focuser.set(Focuser.OPT_FOCUS, self.af_current_pos)

        print("[INIT] Starting Initial AutoFocus...")

        # AutoFocus object.
        # We pass camera=None because frames come from Hailo / GStreamer,
        # and are given directly to stepFocus_hailo().
        self.autofocus = AutoFocus(self.focuser, camera=None)
        self.autofocus.debug = True

        # Only run this autofocus session once
        self.af_done_once = False

        # Initialize incremental autofocus (coarse/fine state machine)
        self.autofocus.startFocus_hailo()

        print("[INIT] Ready. Camera will move RIGHT in ~2 seconds.")
        print("=" * 40 + "\n")


# =====================================================================
# CALLBACK FUNCTION (Runs ~30 times per second)
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    """
    This function is called for each buffer that passes through the pipeline.
    It must be **fast** and must NOT block, otherwise the whole GStreamer
    pipeline will stall.
    """
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Read ROI metadata (currently not used, but kept for future logic)
    roi = hailo.get_roi_from_buffer(buffer)

    # Increase frame counter (used for the 3-second timer)
    user_data.frame_counter += 1

    # Extract the current video frame as a NumPy array
    fmt, w, h = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer, fmt, w, h)
    if frame is None:
        return Gst.PadProbeReturn.OK

    # -----------------------------------------------------------
    # 3-SECOND TIMER: move camera once after ~90 frames
    # -----------------------------------------------------------
    # ~30 FPS → 90 frames ≈ 3 seconds
    if user_data.frame_counter == 90 and not user_data.has_moved_once:
        print("\n>>> [TIMER] 3 Seconds passed! Moving Camera to Right...")
        user_data.focuser.set(Focuser.OPT_MOTOR_X, 0)
        user_data.has_moved_once = True
        print("done")

    # -----------------------------------------------------------
    # INCREMENTAL AUTOFOCUS (non-blocking)
    # -----------------------------------------------------------
    if user_data.is_focusing:
        # Run one autofocus step on this frame
        finished, best_pos = user_data.autofocus.stepFocus_hailo(frame)

        if finished:
            # Autofocus has found a best position
            print(f"!!! [AF-H] FINISHED! Best Focus Position: {best_pos} !!!")
            user_data.is_focusing = False

            # Apply best focus immediately
            user_data.focuser.set(Focuser.OPT_FOCUS, best_pos)
            print(f"focus on {best_pos}")

            # ----------------------------------------------------------------
            # If you still want your original "debug" behaviour:
            # after 10 seconds, print "check" and set focus back to 200.
            # We do this using GLib timeout so we don't block the pipeline.
            # ----------------------------------------------------------------
            '''def _restore_focus():
                print("check")
                user_data.focuser.set(Focuser.OPT_FOCUS, 270)
                return False  # run only once'''

            # Schedule the restore callback 10 seconds from now
          #  GLib.timeout_add_seconds(10, _restore_focus)

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
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass
