#!/usr/bin/env python3
import os
import sys
import time
import argparse
import gi
import cv2
import numpy as np
from contextlib import contextmanager

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

# --- HAILO IMPORTS (From the repo you sent) ---
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad
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
# HELPER: Get Writable Buffer (Aligned with Hailo's Utils)
# =====================================================================
@contextmanager
def get_writable_ndarray(buffer, width, height):
    """
    Maps the GStreamer buffer with WRITE permissions.
    We pass explicit width/height integers because 'get_caps_from_pad'
    already parsed the structure for us.
    """
    # 1. Force the buffer to be writable
    # This acts as a 'smart pointer' to the video memory
    success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

    if not success:
        yield None
        return

    try:
        # 2. Wrap the memory in a NumPy array so OpenCV can use it
        # We assume 3 channels (RGB) which is standard for Hailo pipelines
        ndarray = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        yield ndarray
    finally:
        # 3. CRITICAL: Unmap the buffer so GStreamer can take it back
        buffer.unmap(map_info)


# =====================================================================
# USER APP CLASS
# =====================================================================
class UserApp(app_callback_class):
    def __init__(self):
        super().__init__()

        print("\n[INIT] Initializing PTZ Camera System...")
        try:
            self.focuser = Focuser(1)
            self.focuser.set(Focuser.OPT_MODE, 1)
            time.sleep(0.2)
            self.focuser.set(Focuser.OPT_IRCUT, 0)  # Normal colors
            time.sleep(0.5)

            # Move to Home Position (Center)
            print("[INIT] Moving to Center...")
            self.center_pan = 0
            self.center_tilt = 25
            self.focuser.set(Focuser.OPT_MOTOR_X, self.center_pan)
            self.focuser.set(Focuser.OPT_MOTOR_Y, self.center_tilt)
            time.sleep(1.0)
        except Exception as e:
            print(f"[ERROR] PTZ Init failed: {e}")

        # State Variables
        self.current_pan = 0
        self.current_tilt = 25

        # Auto-Focus State
        self.af_running = True
        self.af_pos = 0
        self.af_step = 20
        self.af_max = 600
        self.af_best_val = 0.0
        self.af_best_pos = 0
        self.af_skip_counter = 0

        self.process_counter = 0
        self.frame_w = None
        self.frame_h = None

        print("[INIT] System Ready. Mode: OSD & Detection.\n")


# =====================================================================
# CALLBACK FUNCTION (The 'User Callback' from the Diagram)
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # 1. Use Hailo's utility to get dimensions
    # This returns: string, int, int (e.g., "RGB", 1280, 720)
    fmt, w, h = get_caps_from_pad(pad)

    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # 2. Open the buffer for WRITING using our helper
    with get_writable_ndarray(buffer, w, h) as frame:
        if frame is None:
            return Gst.PadProbeReturn.OK

        user_data.process_counter += 1

        # -----------------------------------------------------------------
        # A. DRAW STATUS INFO
        # -----------------------------------------------------------------
        status_text = f"Pan: {user_data.current_pan} | Tilt: {user_data.current_tilt}"

        # Black border (Thickness 5)
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 5)
        # Yellow text (Thickness 2)
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)

        # -----------------------------------------------------------------
        # B. AUTO-FOCUS LOGIC
        # -----------------------------------------------------------------
        if user_data.af_running:
            cv2.putText(frame, "Auto-Focusing...", (50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check focus every 3rd frame
            if user_data.process_counter % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()

                if score > user_data.af_best_val:
                    user_data.af_best_val = score
                    user_data.af_best_pos = user_data.af_pos

                user_data.af_pos += user_data.af_step

                if user_data.af_pos <= user_data.af_max:
                    user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_pos)
                else:
                    print(f"[AF] Done. Best Position: {user_data.af_best_pos}")
                    user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                    user_data.af_running = False

            # While focusing, skip detection drawing
            return Gst.PadProbeReturn.OK

        # -----------------------------------------------------------------
        # C. DETECTION LOGIC
        # -----------------------------------------------------------------
        # We read the metadata that the HAILO chip already put on the buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        for det in detections:
            if det.get_label() == "person":
                bbox = det.get_bbox()

                # Convert 0.0-1.0 coords to pixels
                xmin = int(bbox.xmin() * w)
                ymin = int(bbox.ymin() * h)

                # Draw "Person Detected"
                msg = "Person Detected"
                # Border
                cv2.putText(frame, msg, (xmin, ymin - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                # Text
                cv2.putText(frame, msg, (xmin, ymin - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

    print(f"[MAIN] Starting GStreamer Pipeline...")
    print(f"[MAIN] Mode: Detection Only + OSD")

    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()