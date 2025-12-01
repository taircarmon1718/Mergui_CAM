#!/usr/bin/env python3
import os
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import hailo
import numpy as np

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

# PTZ
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from B016712MP.Focuser import Focuser


# =====================================================================
# User App Setup (runs ONCE)
# =====================================================================
class UserApp(app_callback_class):
    def __init__(self):
        super().__init__()

        print("\n========== PTZ INITIALIZATION ==========\n")

        # Create focuser
        self.focuser = Focuser(1)

        # Enable motors
        print("Enabling motors (OPT_MODE = 1)…")
        self.focuser.set(Focuser.OPT_MODE, 1)
        time.sleep(0.5)

        # Disable IR-CUT
        print("Disabling IR-CUT (OPT_IRCUT = 0)…")
        self.focuser.set(Focuser.OPT_IRCUT, 0)
        time.sleep(0.5)

        # Reset chip registers (same as your original code)
        print("Resetting chip registers…")
        try:
            self.focuser.write(self.focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
        except Exception as e:
            print("Chip reset skipped:", e)
        time.sleep(0.5)

        # Initial center-ish movement (like your code)
        print("Moving PTZ to initial position…")
        self.focuser.set(Focuser.OPT_MOTOR_X, 300)
        time.sleep(2)
        self.focuser.set(Focuser.OPT_MOTOR_Y, 25)
        time.sleep(2)

        # Save first frame’s resolution
        self.frame_w = None
        self.frame_h = None

        # Tracking sensitivity
        self.gain_x = 25
        self.gain_y = 18

        print("\n========== PTZ READY ==========\n")


# =====================================================================
# Callback Function — Runs on EVERY inference frame
# =====================================================================
def app_callback(pad, info, user_data: UserApp):

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get frame metadata
    fmt, w, h = get_caps_from_pad(pad)

    # Save resolution on first frame
    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h
        print(f"Camera resolution detected: {w}x{h}")

    # Convert Hailo buffer → numpy RGB
    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Select the largest person bounding box
    best_det = None
    best_area = 0

    for det in detections:
        if det.get_label() != "person":
            continue

        bbox = det.get_bbox()
        xmin = bbox.xmin()
        ymin = bbox.ymin()
        xmax = xmin + bbox.width()
        ymax = ymin + bbox.height()

        area = bbox.width() * bbox.height()
        if area > best_area:
            best_area = area
            best_det = (xmin, ymin, xmax, ymax)

    # If no person → skip
    if best_det is None:
        return Gst.PadProbeReturn.OK

    # Extract the person bbox center
    xmin, ymin, xmax, ymax = best_det
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Center of frame
    fx = user_data.frame_w / 2
    fy = user_data.frame_h / 2

    # Tracking error
    err_x = cx - fx
    err_y = cy - fy

    # Get current PTZ
    pan = user_data.focuser.get(Focuser.OPT_MOTOR_X)
    tilt = user_data.focuser.get(Focuser.OPT_MOTOR_Y)

    # Compute new PTZ
    new_pan = int(pan + (err_x / user_data.frame_w) * user_data.gain_x)
    new_tilt = int(tilt + (err_y / user_data.frame_h) * user_data.gain_y)

    # PTZ limits
    new_pan = max(0, min(350, new_pan))
    new_tilt = max(0, min(180, new_tilt))

    # Apply movement
    user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
    user_data.focuser.set(Focuser.OPT_MOTOR_Y, new_tilt)

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN PROGRAM
# =====================================================================
if __name__ == "__main__":
    print("Loading environment…")

    # Load .env from project root (not inside scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")

    print(f"HAILO_ENV_FILE = {env_file}")
    os.environ["HAILO_ENV_FILE"] = env_file

    # Create user app
    user_data = UserApp()

    # Force input from RPI camera
    app = GStreamerDetectionApp(app_callback, user_data, input="rpi")

    print("\n========== STARTING HAILO PIPELINE ==========\n")
    app.run()
