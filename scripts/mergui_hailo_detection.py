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
# Adds the parent directory to path so we can import the Focuser class
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from B016712MP.Focuser import Focuser


# =====================================================================
# USER APP CLASS: Handles Initialization & State
# =====================================================================
class UserApp(app_callback_class):
    def __init__(self):
        super().__init__()

        print("\n[INIT] Initializing PTZ Camera System...")

        # 1. Initialize Focuser (I2C bus 1)
        self.focuser = Focuser(1)

        # 2. Reset and Setup Camera Hardware
        # Enable motors
        self.focuser.set(Focuser.OPT_MODE, 1)
        time.sleep(0.2)
        # Disable IR Cut filter (better color)
        self.focuser.set(Focuser.OPT_IRCUT, 0)
        time.sleep(0.2)

        # Reset chip register (prevents I2C hangs)
        try:
            self.focuser.write(self.focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
        except Exception as e:
            print(f"[WARN] Chip reset ignored: {e}")
        time.sleep(0.5)

        # 3. Move to Center Position
        print("[INIT] Moving to Home Position (Pan: 0, Tilt: 25)...")
        self.focuser.set(Focuser.OPT_MOTOR_X, 0)  # Pan center
        time.sleep(1.0)
        self.focuser.set(Focuser.OPT_MOTOR_Y, 25)  # Tilt slightly up
        time.sleep(1.0)

        # 4. Autofocus State Variables
        print("[INIT] Preparing Autofocus Logic...")
        self.af_running = True  # Flag: True = doing AF, False = doing Detection
        self.af_pos = 0  # Current lens position (0-1000)
        self.af_step = 20  # Step size for scanning
        self.af_max = 600  # Max focus limit
        self.af_best_val = 0.0  # Best sharpness score found
        self.af_best_pos = 0  # Position where best score was found
        self.af_skip_counter = 0  # Counter to slow down processing for motors

        # Frame dimensions (will be filled in callback)
        self.frame_w = None
        self.frame_h = None

        print("[INIT] System Ready. Waiting for video stream...\n")


# =====================================================================
# CALLBACK FUNCTION: Runs Every Frame
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    # Get the video buffer
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get resolution (only once)
    fmt, w, h = get_caps_from_pad(pad)
    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # -----------------------------------------------------------------
    # PHASE 1: AUTOFOCUS (Runs first, until focus is found)
    # -----------------------------------------------------------------
    if user_data.af_running:
        user_data.af_skip_counter += 1

        # Process only every 3rd frame to give motors time to move
        if user_data.af_skip_counter % 3 != 0:
            return Gst.PadProbeReturn.OK

        # Convert buffer to image (numpy array)
        frame = get_numpy_from_buffer(buffer, fmt, w, h)
        # Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Calculate Sharpness (Laplacian Variance)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check if this is the best focus so far
        if score > user_data.af_best_val:
            user_data.af_best_val = score
            user_data.af_best_pos = user_data.af_pos

        # Move the lens forward
        user_data.af_pos += user_data.af_step

        if user_data.af_pos <= user_data.af_max:
            # Still scanning...
            user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_pos)
        else:
            # Scanning finished!
            print(f"\n[AUTOFOCUS] Complete. Best Position: {user_data.af_best_pos}")
            # Move lens to the winning position
            user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
            # Stop AF phase, Start Detection phase
            user_data.af_running = False

        return Gst.PadProbeReturn.OK

    # -----------------------------------------------------------------
    # PHASE 2: DETECTION & REPORTING (No Tracking Movement)
    # -----------------------------------------------------------------

    # Get detections from Hailo metadata
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    best_person = None
    max_area = 0

    # Filter detections: Find the largest 'person'
    for det in detections:
        if det.get_label() != "person":
            continue

        bbox = det.get_bbox()
        area = bbox.width() * bbox.height()

        if area > max_area:
            max_area = area
            best_person = det

    # If no person found, exit
    if best_person is None:
        return Gst.PadProbeReturn.OK

    # --- Person Found: Get Data ---

    # 1. Get Motor Positions (Read from hardware)
    # This tells us where the camera is currently looking
    current_pan = user_data.focuser.get(Focuser.OPT_MOTOR_X)
    current_tilt = user_data.focuser.get(Focuser.OPT_MOTOR_Y)

    # 2. Get Bounding Box Coordinates (0.0 to 1.0)
    bbox = best_person.get_bbox()
    xmin = bbox.xmin()
    ymin = bbox.ymin()

    # Calculate center just for reference
    cx = xmin + (bbox.width() / 2)
    cy = ymin + (bbox.height() / 2)

    # 3. Print Data to Console
    print(f"[DETECTED] Pan: {current_pan} | Tilt: {current_tilt} || "
          f"BBox Top-Left: ({xmin:.3f}, {ymin:.3f}) | Center: ({cx:.3f}, {cy:.3f})")

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Parse arguments to satisfy GStreamerDetectionApp requirements
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rpi", help="Input source (default: rpi)")
    args, unknown = parser.parse_known_args()

    # Force input to RPi Camera
    args.input = "rpi"

    # Set Hailo Environment Variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")

    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    print(f"[MAIN] Starting Hailo Pipeline on Input: {args.input}")

    # Create User Data Object (Init PTZ)
    user_data = UserApp()

    # Initialize App
    # Pass 'app_callback' and 'user_data' to the Hailo Wrapper
    app = GStreamerDetectionApp(app_callback, user_data)

    # Run Pipeline
    app.run()