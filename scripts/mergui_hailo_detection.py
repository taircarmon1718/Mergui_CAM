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
# Ensures we can import the Focuser class from the parent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from B016712MP.Focuser import Focuser


# =====================================================================
# USER APP CLASS: Handles Hardware Setup
# =====================================================================
class UserApp(app_callback_class):
    def __init__(self):
        super().__init__()

        print("\n[INIT] Initializing PTZ Camera System...")

        # 1. Initialize Motors
        self.focuser = Focuser(1)
        self.focuser.set(Focuser.OPT_MODE, 1)  # Enable motors
        time.sleep(0.2)
        self.focuser.set(Focuser.OPT_IRCUT, 0)  # Set normal colors
        time.sleep(0.2)

        # 2. Reset Chip (Prevent I2C hangs)
        try:
            self.focuser.write(self.focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
        except Exception:
            pass
        time.sleep(0.5)

        # 3. Move to Home Position
        print("[INIT] Moving to Center...")
        self.focuser.set(Focuser.OPT_MOTOR_X, 0)
        self.focuser.set(Focuser.OPT_MOTOR_Y, 25)
        time.sleep(1.0)

        # 4. Auto-Focus Variables
        self.af_running = True
        self.af_pos = 0
        self.af_step = 20
        self.af_max = 600
        self.af_best_val = 0.0
        self.af_best_pos = 0
        self.af_skip_counter = 0

        self.frame_w = None
        self.frame_h = None

        print("[INIT] System Ready.\n")


# =====================================================================
# CALLBACK FUNCTION: Runs Every Frame
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get resolution once
    fmt, w, h = get_caps_from_pad(pad)
    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # Get the image frame (Writable NumPy array)
    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    # -----------------------------------------------------------------
    # PHASE 1: Auto-Focus Logic (Runs first)
    # -----------------------------------------------------------------
    if user_data.af_running:
        # Draw status on screen
        cv2.putText(frame, "Auto-Focusing...", (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        user_data.af_skip_counter += 1
        # Process every 3rd frame to let motors move
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
                print(f"[AF] Focus Found at: {user_data.af_best_pos}")
                user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                user_data.af_running = False

    # -----------------------------------------------------------------
    # PHASE 2: Detection & Drawing
    # -----------------------------------------------------------------

    # Get detections from Hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for det in detections:
        # We only care about persons
        if det.get_label() == "person":
            bbox = det.get_bbox()

            # Convert 0.0-1.0 coordinates to Pixels
            xmin = int(bbox.xmin() * w)
            ymin = int(bbox.ymin() * h)
            xmax = int(bbox.xmax() * w)
            ymax = int(bbox.ymax() * h)

            # Draw Green Box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw Label
            cv2.putText(frame, "Person", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -----------------------------------------------------------------
    # PHASE 3: On-Screen Display (Pan/Tilt Values)
    # -----------------------------------------------------------------

    # Read motor positions
    try:
        pan = user_data.focuser.get(Focuser.OPT_MOTOR_X)
        tilt = user_data.focuser.get(Focuser.OPT_MOTOR_Y)
        status_text = f"Pan: {pan} | Tilt: {tilt}"
    except Exception:
        status_text = "Pan: Err | Tilt: Err"

    # Draw Yellow Text at the top-left
    # (Image, Text, Position, Font, Scale, Color(BGR), Thickness)
    cv2.putText(frame, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rpi", help="Input source")
    args, unknown = parser.parse_known_args()
    args.input = "rpi"

    # Setup Environment Variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")

    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    print(f"[MAIN] Starting GStreamer Pipeline...")

    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[STOP] Stopping...")