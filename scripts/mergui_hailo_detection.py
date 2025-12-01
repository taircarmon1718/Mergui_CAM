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

# PTZ imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from B016712MP.Focuser import Focuser


# =====================================================================
# USER APP — runs ONCE, init PTZ, etc.
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
        time.sleep(0.4)

        # Disable IR CUT
        print("Disabling IR-CUT…")
        self.focuser.set(Focuser.OPT_IRCUT, 0)
        time.sleep(0.4)

        # Reset chip
        print("Resetting chip registers…")
        try:
            self.focuser.write(self.focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
        except Exception as e:
            print("Chip reset skipped:", e)
        time.sleep(0.4)

        # Initial PTZ position
        print("Moving PTZ to initial position…")
        self.focuser.set(Focuser.OPT_MOTOR_X, 300)
        time.sleep(1)
        self.focuser.set(Focuser.OPT_MOTOR_Y, 25)
        time.sleep(1)

        print("\n========== PTZ READY ==========\n")

        # Tracking config
        self.frame_w = None
        self.frame_h = None
        self.gain_x = 25
        self.gain_y = 18


# =====================================================================
# CALLBACK — runs EVERY frame
# =====================================================================
def app_callback(pad, info, user_data: UserApp):

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    fmt, w, h = get_caps_from_pad(pad)
    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h
        print(f"Camera resolution: {w}x{h}")

    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    # get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    best = None
    best_area = 0

    for det in detections:
        if det.get_label() != "person":
            continue

        bbox = det.get_bbox()
        area = bbox.width() * bbox.height()

        if area > best_area:
            best_area = area
            best = det

    # no person
    if best is None:
        return Gst.PadProbeReturn.OK

    bbox = best.get_bbox()
    cx = bbox.xmin() + bbox.width() / 2
    cy = bbox.ymin() + bbox.height() / 2

    fx = user_data.frame_w / 2
    fy = user_data.frame_h / 2

    err_x = cx - fx
    err_y = cy - fy

    # current ptz
    pan = user_data.focuser.get(Focuser.OPT_MOTOR_X)
    tilt = user_data.focuser.get(Focuser.OPT_MOTOR_Y)

    # compute new PTZ
    new_pan = int(pan + (err_x / user_data.frame_w) * user_data.gain_x)
    new_tilt = int(tilt + (err_y / user_data.frame_h) * user_data.gain_y)

    # limits
    new_pan = max(0, min(350, new_pan))
    new_tilt = max(0, min(180, new_tilt))

    user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
    user_data.focuser.set(Focuser.OPT_MOTOR_Y, new_tilt)

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")

    print("Loading environment file:", env_file)
    os.environ["HAILO_ENV_FILE"] = env_file

    # 🔥 CRITICAL — FORCE CAMERA INPUT
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"
    print(">>> Forced input source =", os.getenv("HAILO_PIPELINE_INPUT"))

    # create PTZ + state object
    user_data = UserApp()

    print("\n========== STARTING HAILO RPI CAMERA PIPELINE ==========\n")
    app = GStreamerDetectionApp(app_callback, user_data)

    app.run()
