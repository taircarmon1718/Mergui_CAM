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

# --- HAILO IMPORTS ---
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
# HELPER: Get Writable Buffer (Corrected Version)
# =====================================================================
@contextmanager
def get_writable_ndarray(buffer, width, height):
    """
    Maps GStreamer buffer with WRITE permissions using known width/height.
    Fixes the 'AttributeError' by not relying on caps structure parsing.
    """
    # Try to make the buffer writable if possible (fixes GStreamer CRITICAL warnings)
    if not buffer.is_writable():
        try:
            # We cannot easily replace the buffer in a probe, so we force map.
            # In some GStreamer versions, this might still warn, but often works.
            pass
        except Exception:
            pass

    # Request WRITE access
    success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

    if not success:
        # If write mapping fails, we cannot draw
        yield None
        return

    try:
        # Create NumPy array (Height, Width, 3 Channels for RGB)
        ndarray = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        yield ndarray
    finally:
        # Unmap is crucial to prevent memory leaks/freezes
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

            print("[INIT] Moving to Center...")
            self.center_pan = 0
            self.center_tilt = 25
            self.focuser.set(Focuser.OPT_MOTOR_X, self.center_pan)
            self.focuser.set(Focuser.OPT_MOTOR_Y, self.center_tilt)
            time.sleep(1.0)
        except Exception as e:
            print(f"[ERROR] PTZ Init failed: {e}")

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

        print("[INIT] System Ready. Tracking is DISABLED.\n")


# =====================================================================
# CALLBACK FUNCTION
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get width and height from the pad caps
    # fmt is a string (e.g. "RGB"), w and h are integers
    fmt, w, h = get_caps_from_pad(pad)

    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # --- FIXED CALL: Pass w and h directly (Integers), NOT 'fmt' ---
    with get_writable_ndarray(buffer, w, h) as frame:
        if frame is None:
            return Gst.PadProbeReturn.OK

        user_data.process_counter += 1

        # -----------------------------------------------------------------
        # 1. DRAW INFO (Pan/Tilt)
        # -----------------------------------------------------------------
        status_text = f"Pan: {user_data.current_pan} | Tilt: {user_data.current_tilt}"

        # Black border
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 5)
        # Yellow text
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)

        # -----------------------------------------------------------------
        # 2. AUTO-FOCUS LOGIC
        # -----------------------------------------------------------------
        if user_data.af_running:
            cv2.putText(frame, "Auto-Focusing...", (50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
                    print(f"[AF] Done. Best: {user_data.af_best_pos}")
                    user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                    user_data.af_running = False

            # While focusing, return here (don't draw boxes yet)
            return Gst.PadProbeReturn.OK

        # -----------------------------------------------------------------
        # 3. DETECTION LOGIC
        # -----------------------------------------------------------------
        # Read Hailo Metadata
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        for det in detections:
            if det.get_label() == "person":
                bbox = det.get_bbox()

                # Convert normalized coords to pixels
                xmin = int(bbox.xmin() * w)
                ymin = int(bbox.ymin() * h)

                # Draw "Person Detected" Text
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
    print(f"[MAIN] Mode: Detection Only (No Tracking)")

    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()