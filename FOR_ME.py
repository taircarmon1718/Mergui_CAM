import sys
import time
import cv2
from flask import Flask, Response
from picamera2 import Picamera2

# ============================================================
# Add project root so PTZ library can be imported
# ============================================================
sys.path.append("/Users/taircarmon/Desktop/Mergui_CAM")

from B016712MP.Focuser import Focuser
from B016712MP.AutoFocus import AutoFocus

# ============================================================
# Flask app
# ============================================================
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
focuser.set(Focuser.OPT_MODE, 1)  # Enable motors
time.sleep(0.5)

print("Disabling IR-CUT...")
focuser.set(Focuser.OPT_IRCUT, 0)
time.sleep(0.5)

# ============================================================
# Initial PTZ movement
# ============================================================
print("Initial pan/tilt movement...")

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
# Streaming generator
# ============================================================
def generate_frames():
    while True:
        frame = cam.capture_array()

        # Rotate (same as your imshow code)
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)

        # Encode frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if not ret:
            continue

        # Return MJPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() +
               b'\r\n')


@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# Main entry
# ============================================================
if __name__ == "__main__":
    print("\n=====================================")
    print("Mergui Camera Stream Running!")
    print("Open this URL in your browser:")
    print("    http://<IP_OF_PI>:8000/video")
    print("=====================================\n")

    app.run(host="0.0.0.0", port=8000, debug=False)

    # Cleanup after exit
    cam.stop()
    cam.close()
    focuser.set(Focuser.OPT_MODE, 0)
    time.sleep(0.5)
    focuser.write(focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
    print("Shutdown complete.")
