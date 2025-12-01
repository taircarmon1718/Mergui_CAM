#!/usr/bin/env python3
import os
import time
import sys
import gi
import cv2
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

# Hailo helper imports
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
    get_default_parser,
)

from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import (
    GStreamerDetectionApp,
)

# PTZ imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from B016712MP.Focuser import Focuser


# =====================================================================
# USER APP - STATE MANAGEMENT
# =====================================================================
class UserApp(app_callback_class):
    def __init__(self):
        super().__init__()

        print("[INFO] PTZ Initialization started...")
        self.focuser = Focuser(1)

        # Init Motors
        self.focuser.set(Focuser.OPT_MODE, 1)
        self.focuser.set(Focuser.OPT_IRCUT, 0)

        # Reset position
        self.focuser.set(Focuser.OPT_MOTOR_X, 300)
        self.focuser.set(Focuser.OPT_MOTOR_Y, 25)

        # --- Autofocus State Variables ---
        self.perform_autofocus = True  # Flag to run focus once
        self.focus_pos = 0  # Current lens position
        self.focus_step = 20  # Step size for scanning
        self.focus_max = 600  # Max limit for focus motor
        self.best_focus_val = 0.0  # Highest sharpness score found
        self.best_focus_pos = 0  # Position of highest sharpness
        self.frame_counter = 0  # Frame skipper for motor stability

        # --- Tracking Variables ---
        self.frame_w = None
        self.frame_h = None
        self.gain_x = 25
        self.gain_y = 18

        print("[INFO] PTZ Ready. Waiting for pipeline...")


# =====================================================================
# CALLBACK - RUNS EVERY FRAME
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get video dimensions
    fmt, w, h = get_caps_from_pad(pad)
    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # ==========================================================
    # PHASE 1: AUTOFOCUS (Runs once at startup)
    # ==========================================================
    if user_data.perform_autofocus:
        user_data.frame_counter += 1

        # Process every 4th frame to allow lens motor to settle
        if user_data.frame_counter % 4 != 0:
            return Gst.PadProbeReturn.OK

        # Convert buffer to numpy array for OpenCV
        frame = get_numpy_from_buffer(buffer, fmt, w, h)

        # Calculate Sharpness (Laplacian Variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # print(f"[FOCUS] Pos: {user_data.focus_pos} | Sharpness: {score:.2f}")

        # Keep track of best focus
        if score > user_data.best_focus_val:
            user_data.best_focus_val = score
            user_data.best_focus_pos = user_data.focus_pos

        # Move lens
        user_data.focus_pos += user_data.focus_step

        # Check if scan is complete
        if user_data.focus_pos <= user_data.focus_max:
            user_data.focuser.set(Focuser.OPT_FOCUS, user_data.focus_pos)
        else:
            # Apply best focus found and disable flag
            print(f"[INFO] Autofocus Complete. Best Position: {user_data.best_focus_pos}")
            user_data.focuser.set(Focuser.OPT_FOCUS, user_data.best_focus_pos)
            user_data.perform_autofocus = False  # STOP AUTOFOCUS PERMANENTLY

        return Gst.PadProbeReturn.OK

    # ==========================================================
    # PHASE 2: DETECTION & TRACKING (Runs after focus is done)
    # ==========================================================

    # Get Hailo detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    best_target = None
    max_area = 0

    # Filter for 'person'
    for det in detections:
        if det.get_label() != "person":
            continue

        bbox = det.get_bbox()
        area = bbox.width() * bbox.height()

        if area > max_area:
            max_area = area
            best_target = det

    if best_target is None:
        return Gst.PadProbeReturn.OK

    # Calculate center error
    bbox = best_target.get_bbox()
    cx = bbox.xmin() + bbox.width() / 2
    cy = bbox.ymin() + bbox.height() / 2

    # Error relative to center (0.5, 0.5)
    err_x = (cx - 0.5)
    err_y = (cy - 0.5)

    # Get current motor position
    current_pan = user_data.focuser.get(Focuser.OPT_MOTOR_X)
    current_tilt = user_data.focuser.get(Focuser.OPT_MOTOR_Y)

    # Calculate new position (PID-like proportional control)
    # Note: Direction (+/-) depends on camera mounting. Flip signs if moving wrong way.
    new_pan = int(current_pan - (err_x * user_data.gain_x))
    new_tilt = int(current_tilt - (err_y * user_data.gain_y))

    # Safety Limits
    new_pan = max(0, min(1000, new_pan))
    new_tilt = max(0, min(1000, new_tilt))

    # Move Motors
    user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
    user_data.focuser.set(Focuser.OPT_MOTOR_Y, new_tilt)

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # 1. Parse Arguments
    parser = get_default_parser()
    args = parser.parse_args()

    # 2. Force Input to RPi Camera
    args.input = "rpi"

    # 3. Environment Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")

    print(f"[INFO] Loading env: {env_file}")
    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    # 4. Initialize User Data (PTZ)
    user_data = UserApp()

    # 5. Run App
    print("[INFO] Starting Pipeline...")
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()