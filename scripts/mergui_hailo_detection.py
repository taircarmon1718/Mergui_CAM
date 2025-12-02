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

        # --- הגדרות חומרה ומנועים ---
        print("\n[INIT] Initializing PTZ Camera System...")
        self.focuser = Focuser(1)
        self.focuser.set(Focuser.OPT_MODE, 1)
        time.sleep(0.2)
        self.focuser.set(Focuser.OPT_IRCUT, 0)
        time.sleep(0.2)

        # איפוס צ'יפ למניעת תקיעות
        try:
            self.focuser.write(self.focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
        except Exception:
            pass
        time.sleep(0.5)

        # מיקום התחלתי
        print("[INIT] Moving to Home Position...")
        self.focuser.set(Focuser.OPT_MOTOR_X, 0)
        time.sleep(1.0)
        self.focuser.set(Focuser.OPT_MOTOR_Y, 25)
        time.sleep(1.0)

        # משתני פוקוס
        self.af_running = True
        self.af_pos = 0
        self.af_step = 20
        self.af_max = 600
        self.af_best_val = 0.0
        self.af_best_pos = 0
        self.af_skip_counter = 0

        self.frame_w = None
        self.frame_h = None

        print("[INIT] Ready.\n")


# =====================================================================
# CALLBACK FUNCTION
# =====================================================================
def app_callback(pad, info, user_data: UserApp):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    fmt, w, h = get_caps_from_pad(pad)
    if user_data.frame_w is None:
        user_data.frame_w = w
        user_data.frame_h = h

    # המרת הבאפר לתמונה שאפשר לערוך (numpy array)
    # הערה: השינויים שנבצע כאן על 'frame' יוצגו על המסך
    frame = get_numpy_from_buffer(buffer, fmt, w, h)

    # -----------------------------------------------------------------
    # קריאת נתוני המנועים (בשביל התצוגה)
    # -----------------------------------------------------------------
    # אנחנו שמים את זה ב-try כי לפעמים התקשורת I2C נכשלת לרגע
    try:
        pan = user_data.focuser.get(Focuser.OPT_MOTOR_X)
        tilt = user_data.focuser.get(Focuser.OPT_MOTOR_Y)
        pan_str = str(pan)
        tilt_str = str(tilt)
    except:
        pan_str = "Err"
        tilt_str = "Err"

    # -----------------------------------------------------------------
    # שלב 1: פוקוס אוטומטי
    # -----------------------------------------------------------------
    if user_data.af_running:
        # כתיבה על המסך שאנחנו בפוקוס
        cv2.putText(frame, f"Auto-Focusing: {user_data.af_pos}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        user_data.af_skip_counter += 1
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
                print(f"[AF] Complete. Best: {user_data.af_best_pos}")
                user_data.focuser.set(Focuser.OPT_FOCUS, user_data.af_best_pos)
                user_data.af_running = False

    # -----------------------------------------------------------------
    # שלב 2: זיהוי והדפסה על המסך (OSD)
    # -----------------------------------------------------------------

    # 1. קבלת זיהויים מה-Hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    best_person = None
    max_area = 0

    for det in detections:
        if det.get_label() == "person":
            bbox = det.get_bbox()
            area = bbox.width() * bbox.height()
            if area > max_area:
                max_area = area
                best_person = det

    # 2. ציור מידע על המסך (OSD) - החלק שביקשת
    # צבע צהוב: (0, 255, 255)

    # כותרת למעלה
    cv2.putText(frame, f"Pan: {pan_str} | Tilt: {tilt_str}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # אם זוהה בן אדם - נצייר עליו מידע
    if best_person:
        bbox = best_person.get_bbox()
        # המרת קואורדינטות יחסיות (0-1) לפיקסלים
        xmin = int(bbox.xmin() * w)
        ymin = int(bbox.ymin() * h)

        # ציור עיגול בפינה של הבן אדם
        cv2.circle(frame, (xmin, ymin), 10, (0, 255, 0), -1)

        # כתיבת טקסט ליד הבן אדם
        msg = f"Person detected"
        cv2.putText(frame, msg, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # הדפסה גם לקונסול
        print(f"Pan: {pan_str} | Tilt: {tilt_str} | Person at X:{xmin} Y:{ymin}")

    return Gst.PadProbeReturn.OK


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rpi", help="Input source")
    args, unknown = parser.parse_known_args()
    args.input = "rpi"

    # Set Environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_file = os.path.join(project_root, ".env")
    os.environ["HAILO_ENV_FILE"] = env_file
    os.environ["HAILO_PIPELINE_INPUT"] = "rpi"

    # Start App
    user_data = UserApp()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()