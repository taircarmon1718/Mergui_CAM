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
# We use the STANDARD Hailo functions. No custom hacks.
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
        try:
            self.focuser = Focuser(1)
            self.focuser.set(Focuser.OPT_MODE, 1)
            time.sleep(0.2)
            self.focuser.set(Focuser.OPT_IRCUT, 0)
            time.sleep(0.5)

            # Move to Home
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
        self.step_size = 50  # How much to move per key press

        # Auto-Focus State
        self.af_running = True
        self.af_pos = 0
        self.af_step = 20
        self.af_max = 600
        self.af_best_val = 0.0
        self.af_best_pos = 0
        self.af_skip_counter = 0

        self.frame_w = None
        self.frame_h = None

        print("\n" + "=" * 40)
        print(" CONTROL INSTRUCTIONS:")
        print(" 1. Click on the video window to focus it.")
        print(" 2. Press [A] to Move Left")
        print(" 3. Press [D] to Move Right")
        print(" 4. Press [S] to Center")
        print(" 5. Press [Q] to Quit")
        print("=" * 40 + "\n")


# =====================================================================
# CALLBACK FUNCTION
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # 1. Get Image Dimensions (Standard Way)
    fmt, w, h = get_caps_from_pad(pad)

    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # 2. Get the image SAFELY (Read-Only)
    # This creates a copy that is safe to touch
    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    # 3. Convert to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # -----------------------------------------------------------------
    # DRAWING ON THE COPY (Safe!)
    # -----------------------------------------------------------------
    status_text = f"Pan: {user_data.current_pan} | Tilt: {user_data.current_tilt}"
    cv2.putText(frame_bgr, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2)

    # -----------------------------------------------------------------
    # AUTO-FOCUS LOGIC
    # -----------------------------------------------------------------
    if user_data.af_running:
        cv2.putText(frame_bgr, "Auto-Focusing...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

        user_data.af_skip_counter += 1
        if user_data.af_skip_counter % 3 == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if score > user_data.af_best_val:
                user_data.af_best_val = score
                user_data.af_best_pos = user_data.af_pos

            user_data.af_pos += user_data.af_step

            if user_data.af_pos <= user_data.af_max:
                user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_pos)
            else:
                print(f"[AF] COMPLETE. Best Position: {user_data.af_best_pos}")
                user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                user_data.af_running = False

    # -----------------------------------------------------------------
    # MANUAL CONTROL & DISPLAY
    # -----------------------------------------------------------------

    # Show the video window using OpenCV
    # This works because we are showing the COPY, not the GStreamer buffer
    cv2.imshow("Hailo Camera Control", frame_bgr)

    # Wait 1ms for a key press
    key = cv2.waitKey(1) & 0xFF

    # Logic to move motors based on key
    target_pan = user_data.current_pan

    if key == ord('a'):  # Left
        target_pan = max(0, user_data.current_pan - user_data.step_size)
        print("<< LEFT")

    elif key == ord('d'):  # Right
        target_pan = min(1000, user_data.current_pan + user_data.step_size)
        print("RIGHT >>")

    elif key == ord('s'):  # Center
        target_pan = 0
        print("|| CENTER")

    elif key == ord('q'):  # Quit
        print("Quitting...")
        os._exit(0)

    # Execute Movement if changed
    if target_pan != user_data.current_pan:
        user_data.focuser.set(Focuser.OPT_MOTOR_X, target_pan)
        user_data.current_pan = target_pan
        print(f"--> MOVED: Pan={user_data.current_pan}")

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rpi", help="Input source")
    # We force the GStreamer sink to be 'fakesink' because we are using
    # our own cv2.imshow window above. This prevents double windows.
    parser.add_argument("--video-sink", default="fakesink", help="Video sink")

    args, unknown = parser.parse_known_args()
    args.input = "rpi"
    args.video_sink = "fakesink"

    # Setup Environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")
    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    print(f"[MAIN] Starting Safe Pipeline...")

    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass