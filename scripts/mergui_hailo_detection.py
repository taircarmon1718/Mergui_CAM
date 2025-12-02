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
# HELPER: Get Writable Buffer (FIXED VERSION)
# =====================================================================
@contextmanager
def get_writable_ndarray(buffer, width, height):
    """
    Context manager to map the GStreamer buffer with WRITE permissions.
    We pass width and height explicitly to avoid object parsing errors.
    """
    # Request WRITE access.
    # Note: If you see 'GStreamer-CRITICAL' warnings in the terminal,
    # it is normal for this kind of direct-drawing hack.
    success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

    if not success:
        # If mapping fails, we cannot draw, so we yield None
        yield None
        return

    try:
        # Create a NumPy array backed by the buffer memory
        # We assume 3 channels (RGB)
        ndarray = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        yield ndarray
    finally:
        # Crucial: Unmap the buffer so GStreamer can use it again
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

        # Store current position (static)
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

    # Get width (w) and height (h) as Integers directly
    fmt, w, h = get_caps_from_pad(pad)

    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # --- FIXED CALL: Pass 'w' and 'h' directly ---
    with get_writable_ndarray(buffer, w, h) as frame:
        if frame is None:
            return Gst.PadProbeReturn.OK

        user_data.process_counter += 1

        # -----------------------------------------------------------------
        # 1. DRAW INFO (Always runs)
        # -----------------------------------------------------------------
        status_text = f"Pan: {user_data.current_pan} | Tilt: {user_data.current_tilt}"

        # Black border for contrast
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 5)
        # Yellow text
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)

        # -----------------------------------------------------------------
        # 2. AUTO-FOCUS LOGIC (Runs once at startup)
        # -----------------------------------------------------------------
        if