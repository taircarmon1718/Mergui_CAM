"""
===========================================================================
Mergui PTZ Camera Preview Script (Picamera2 + ArduCam PTZ + Autofocus)
===========================================================================

This script initializes and tests the complete PTZ camera pipeline on the
Raspberry Pi using the ArduCam PTZ module and Picamera2.

The script performs the following steps:

1. Load Custom PTZ Modules
   • Adds the project directory to sys.path so Python can import:
       - Focuser  (PTZ motor controller)
       - AutoFocus (lens focusing algorithm)

2. Camera Initialization (Picamera2)
   • Configures video mode at 640×360, RGB888
   • Starts camera stream
   • Prepares frames for OpenCV-based preview

3. PTZ (Pan–Tilt–Zoom) Initialization
   • Enables PTZ motors
   • Disables IR-CUT for normal color preview
   • Performs an initial pan + tilt movement to verify motor response

4. Autofocus Procedure
   • Runs the AutoFocus.startFocus2() routine
   • Sweeps lens positions and computes sharpness values
   • Chooses the best focal position and applies it
   • Prints final focus index and score

5. Live Preview Window
   • Continuously captures frames from Picamera2
   • Rotates the frame to match the PTZ orientation
   • Displays a real-time OpenCV preview window

   **Additional overlay drawn on the frame:**
       • A fixed rectangle in the center of the screen
       • On-screen text showing the rectangle's (x, y) coordinates
         relative to the current frame size



6. Safe Shutdown
   • Stops camera stream and closes resources
   • Waits for all PTZ operations to finish
   • Disables motor mode
   • Resets internal PTZ chip registers
   • Closes all OpenCV windows

This script is intended as a base for:
   ▸ PTZ auto-tracking system
   ▸ YOLO detection overlay
   ▸ MediaMTX / RTSP streaming pipeline
   ▸ Halo AI integration

===========================================================================
"""
import sys
import time
import threading
import cv2
import numpy as np
from picamera2 import Picamera2
# ============================================================
# Add PTZ library path
# ============================================================
sys.path.append("/home/tair/Desktop/Mergui_CAM")

from B016712MP.Focuser import Focuser
from B016712MP.AutoFocus import AutoFocus
# ============================================================
# Camera Setup
# ============================================================
cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (640, 360), "format": "RGB888"}
))
cam.start()
time.sleep(2)
# ============================================================
# PTZ / Focuser Setup
# ============================================================
focuser = Focuser(1)
focuser.set(Focuser.OPT_MODE, 1)
time.sleep(0.5)
focuser.set(Focuser.OPT_IRCUT, 0)
time.sleep(0.5)
print("Initial PTZ movement...")
focuser.set(Focuser.OPT_MOTOR_X, 300)
time.sleep(2)
focuser.set(Focuser.OPT_MOTOR_Y, 25)
time.sleep(2)
print("Movement complete.")
# ============================================================
# Auto-Focus
# ============================================================
print("Starting AutoFocus...")
auto_focus = AutoFocus(focuser, cam)
auto_focus.debug = True
max_index, max_value = auto_focus.startFocus2()
print(f"Autofocus completed: index={max_index}, value={max_value}")
time.sleep(1)
# ============================================================
# Frame Capture Thread (FAST & SMOOTH)
# ============================================================
latest_frame = None
lock = threading.Lock()
def capture_thread():
    global latest_frame
    while True:
        frame = cam.capture_array()
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)

        with lock:
            latest_frame = frame
threading.Thread(target=capture_thread, daemon=True).start()
# ============================================================
# Main Preview Loop
# ============================================================
try:
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Draw overlay rectangle
        height, width, _ = frame.shape
        rect_w, rect_h = 200, 100
        top_left = ((width - rect_w) // 2, (height - rect_h) // 2)
        bottom_right = ((width + rect_w) // 2, (height + rect_h) // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Draw coordinates text
        coord_text = f"({top_left[0]}, {top_left[1]})"
        cv2.putText(frame, coord_text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Mergui PTZ Camera Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
# ============================================================
# Shutdown / Reset Everything
# ============================================================
cam.stop()
focuser.waitIdle()
focuser.set(Focuser.OPT_MODE, 0)  # Disable motors
focuser.resetAll()
cv2.destroyAllWindows()
print("Shutdown complete.")



