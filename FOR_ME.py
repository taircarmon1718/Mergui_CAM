import sys
import time
import cv2
import threading
from flask import Flask, Response
from picamera2 import Picamera2

# ============================================================
# Add PTZ library path
# ============================================================
sys.path.append("/Users/taircarmon/Desktop/Mergui_CAM")

from B016712MP.Focuser import Focuser
from B016712MP.AutoFocus import AutoFocus

app = Flask(__name__)

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
time.sleep(1)
focuser.set(Focuser.OPT_MOTOR_Y, 25)
time.sleep(1)

print("Starting AutoFocus...")
auto_focus = AutoFocus(focuser, cam)
auto_focus.debug = False
auto_focus.startFocus2()
time.sleep(0.5)

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

        time.sleep(0.01)  # limits to ~100 FPS input

t = threading.Thread(target=capture_thread, daemon=True)
t.start()

# ============================================================
# Flask Streaming (optimized MJPEG)
# ============================================================
def generate_stream():
    global latest_frame

    while True:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            continue

        # JPEG encode — quality optimized
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

        time.sleep(0.03)  # OUTPUT FPS ~ 30 FPS


@app.route("/video")
def video():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# Run Flask
# ============================================================
if __name__ == "__main__":
    print("\n=====================================")
    print("Improved Mergui Camera Stream Running!")
    print("Open:")
    print("    http://<IP>:8000/video")
    print("=====================================")

    app.run(host="0.0.0.0", port=8000, debug=False)
