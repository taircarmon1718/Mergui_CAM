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

        print("\n[INIT] Initializing PTZ Camera System...")
        self.focuser = Focuser(1)
        self.focuser.set(Focuser.OPT_MODE, 1)
        time.sleep(0.2)
        self.focuser.set(Focuser.OPT_IRCUT, 0)
        time.sleep(0.5)

        # Move to Center
        print("[INIT] Moving to Center...")
        self.center_pan = 0
        self.center_tilt = 25
        self.focuser.set(Focuser.OPT_MOTOR_X, self.center_pan)
        self.focuser.set(Focuser.OPT_MOTOR_Y, self.center_tilt)
        time.sleep(1.0)

        # We keep track of position in variables to avoid reading from hardware (fixes lag)
        self.current_pan = self.center_pan
        self.current_tilt = self.center_tilt

        # Auto-Focus variables
        self.af_running = True
        self.af_pos = 0
        self.af_step = 20
        self.af_max = 600
        self.af_best_val = 0.0
        self.af_best_pos = 0
        self.af_skip_counter = 0

        # Tracking variables
        self.gain_x = 15
        self.gain_y = 10
        self.process_skip = 0

        self.frame_w = None
        self.frame_h = None

        print("[INIT] Ready.\n")


# =====================================================================
# CALLBACK FUNCTION
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    fmt, w, h = get_caps_from_pad(pad)
    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # Get the image to draw on
    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    # -----------------------------------------------------------------
    # PHASE 1: Auto-Focus
    # -----------------------------------------------------------------
    if user_data.af_running:
        user_data.af_skip_counter += 1
        if user_data.af_skip_counter % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if score > user_data.af_best_val:
                user_data.af_best_val = score
                user_data.af_best_pos = user_data.af_pos

            user_data.af_pos += user_data.af_step

            if user_data.af_pos <= user_data.af_max:
                user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_pos)
            else:
                print(f"[AF] Done. Best: {user_data.af_best_pos}")
                user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                user_data.af_running = False

        # Draw AF status
        cv2.putText(frame, "Auto-Focusing...", (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return Gst.PadProbeReturn.OK

    # -----------------------------------------------------------------
    # PHASE 2: Detection & Tracking
    # -----------------------------------------------------------------

    # Optimization: Run tracking logic only every 2nd frame
    user_data.process_skip += 1

    # Always DRAW the text (using saved variables) - This is fast!
    cv2.putText(frame, f"Pan: {user_data.current_pan} | Tilt: {user_data.current_tilt}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    # Only do the heavy lifting (detection/movement) every few frames
    if user_data.process_skip % 2 != 0:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    best_person = None
    max_area = 0

    for det in detections:
        if det.get_label() == "person":
            bbox = det.get_bbox()
            area = bbox.width() * bbox.height()
            if area > max_area:
                max_area = area
                best_person = det

    if best_person:
        bbox = best_person.get_bbox()
        cx = bbox.xmin() + (bbox.width() / 2)
        cy = bbox.ymin() + (bbox.height() / 2)

        # Calculate Error
        err_x = cx - 0.5
        err_y = cy - 0.5

        # Only move if error is significant (prevent jitter)
        if abs(err_x) > 0.04 or abs(err_y) > 0.04:
            # Calculate new position
            new_pan = int(user_data.current_pan - (err_x * user_data.gain_x))
            new_tilt = int(user_data.current_tilt - (err_y * user_data.gain_y))

            # Clamp limits
            new_pan = max(0, min(1000, new_pan))
            new_tilt = max(0, min(1000, new_tilt))

            # Move Motor
            user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
            user_data.focuser.set(Focuser.OPT_MOTOR_Y, new_tilt)

            # Update variables for display
            user_data.current_pan = new_pan
            user_data.current_tilt = new_tilt

            # Draw "Tracking" indicator
            cv2.putText(frame, "TRACKING", (int(bbox.xmin() * w), int(bbox.ymin() * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN
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

    print(f"[MAIN] Starting GStreamer Pipeline...")

    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()