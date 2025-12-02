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
        self.step_size = 50  # Movement speed

        # Create a dummy image for the Controller Window
        self.control_image = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(self.control_image, "CLICK ME TO CONTROL", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.control_image, "A: Left | D: Right", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(self.control_image, "S: Center | Q: Quit", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

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

        print("\n" + "=" * 40)
        print(" INSTRUCTIONS:")
        print(" 1. Two windows will open.")
        print(" 2. Watch the video in the 'Hailo' window.")
        print(" 3. CLICK on the small black 'REMOTE CONTROL' window.")
        print(" 4. Use A / D / S keys to move.")
        print("=" * 40 + "\n")


# =====================================================================
# CALLBACK FUNCTION
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.process_counter += 1

    # 1. Show the "Remote Control" window (Dummy window)
    # This window exists ONLY to capture keyboard presses safely
    cv2.imshow("Remote Control", user_data.control_image)

    # Read Keyboard (1ms delay)
    key = cv2.waitKey(1) & 0xFF

    # 2. Movement Logic
    target_pan = user_data.current_pan

    if key == ord('a'):  # Left
        target_pan = max(0, user_data.current_pan - user_data.step_size)
        print(f"<< LEFT ({target_pan})")

    elif key == ord('d'):  # Right
        target_pan = min(1000, user_data.current_pan + user_data.step_size)
        print(f"RIGHT >> ({target_pan})")

    elif key == ord('s'):  # Center
        target_pan = 0
        print("|| CENTER")

    elif key == ord('q'):  # Quit
        os._exit(0)

    # Apply Movement
    if target_pan != user_data.current_pan:
        user_data.focuser.set(Focuser.OPT_MOTOR_X, target_pan)
        user_data.current_pan = target_pan

    # 3. Auto-Focus Logic (Read-Only from video)
    if user_data.af_running:
        if user_data.process_counter % 3 == 0:
            fmt, w, h = get_caps_from_pad(pad)
            frame = get_numpy_from_buffer(buffer, fmt, w, h)

            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()

                if score > user_data.af_best_val:
                    user_data.af_best_val = score
                    user_data.af_best_pos = user_data.af_pos

                user_data.af_pos += user_data.af_step

                if user_data.af_pos <= user_data.af_max:
                    user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_pos)
                    print(f"[AF] Scanning... {user_data.af_pos}")
                else:
                    print(f"[AF] DONE. Best: {user_data.af_best_pos}")
                    user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                    user_data.af_running = False

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rpi", help="Input source")
    args, unknown = parser.parse_known_args()
    args.input = "rpi"

    # Setup Environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")
    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    print(f"[MAIN] Starting Pipeline...")

    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        pass