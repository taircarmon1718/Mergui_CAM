#!/usr/bin/env python3
import os
import cv2
import numpy as np
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo

# Hailo helpers
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    app_callback_class
)
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import (
    GStreamerDetectionApp
)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# PTZ imports
from B016712MP.Focuser import Focuser
from B016712MP.AutoFocus import AutoFocus


# =====================================================================
# User callback class (runs ONCE)
# =====================================================================
class UserApp(app_callback_class):
    def __init__(self):
        super().__init__()

        # PTZ setup
        self.focuser = Focuser(1)
        self.focuser.set(Focuser.OPT_MODE, 1)

        # Small tilt offset so camera starts centered
        self.focuser.set(Focuser.OPT_MOTOR_Y, 25)

        # Autofocus
        print("Starting autofocus…")
        try:
            af = AutoFocus(self.focuser, None)
            af.startFocus2()
        except Exception as e:
            print("Autofocus skipped:", e)

        # Track window width/height updated later
        self.frame_width = None
        self.frame_height = None

        # PTZ movement gain (controls speed)
        self.gain_x = 30
        self.gain_y = 20


# =====================================================================
# Callback — runs on EVERY frame from Hailo pipeline
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get caps to decode size + format
    format, width, height = get_caps_from_pad(pad)
    if user_data.frame_width is None:
        user_data.frame_width = width
        user_data.frame_height = height

    # Convert buffer → numpy frame
    frame = get_numpy_from_buffer(buffer, format, width, height)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Track biggest person
    best_det = None
    best_area = 0

    for d in detections:
        label = d.get_label()
        if label != "person":
            continue

        bbox = d.get_bbox()
        xmin = bbox.xmin()
        ymin = bbox.ymin()
        xmax = xmin + bbox.width()
        ymax = ymin + bbox.height()

        area = bbox.width() * bbox.height()
        if area > best_area:
            best_area = area
            best_det = (xmin, ymin, xmax, ymax)

    # If no person — show frame only
    if best_det is None:
        cv2.imshow("MERGUI + HAILO", frame)
        cv2.waitKey(1)
        return Gst.PadProbeReturn.OK

    # Draw bbox
    xmin, ymin, xmax, ymax = best_det
    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    # Compute center of detection
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Compute center of frame
    fx = user_data.frame_width / 2
    fy = user_data.frame_height / 2

    # Tracking error
    err_x = cx - fx
    err_y = cy - fy

    # Read current PTZ position
    pan = user_data.focuser.get(Focuser.OPT_MOTOR_X)
    tilt = user_data.focuser.get(Focuser.OPT_MOTOR_Y)

    # Move PTZ to reduce error
    new_pan = int(pan + (err_x / user_data.frame_width) * user_data.gain_x)
    new_tilt = int(tilt + (err_y / user_data.frame_height) * user_data.gain_y)

    # Clamp limits
    new_pan = max(0, min(350, new_pan))
    new_tilt = max(0, min(180, new_tilt))

    user_data.focuser.set(Focuser.OPT_MOTOR_X, new_pan)
    user_data.focuser.set(Focuser.OPT_MOTOR_Y, new_tilt)

    # Display
    cv2.putText(frame, f"Pan:{new_pan} Tilt:{new_tilt}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("MERGUI + HAILO", frame)
    cv2.waitKey(1)

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    # Load .env (must include MODEL_PATH)
    project_root = os.path.abspath(os.path.dirname(__file__))
    env_file = os.path.join(project_root, ".env")
    os.environ["HAILO_ENV_FILE"] = env_file

    # Create callback object
    user_data = UserApp()

    # Start detection
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
