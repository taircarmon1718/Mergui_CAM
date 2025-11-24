import sys
import time
import cv2
from picamera2 import Picamera2

# ============================================================
# Add project root so PTZ library can be imported
# ============================================================
sys.path.append("/Users/taircarmon/Desktop/Mergui_CAM")

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
# Live Preview Loop (NO TRACKING)
# ============================================================
print("Starting live preview (press 'q' to quit)...")

while True:
    frame = cam.capture_array()

    # Rotate for correct orientation
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame, 1)

    cv2.imshow("Mergui Camera Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# Shutdown / Reset Everything
# ============================================================
print("Stopping camera and resetting PTZ...")

cam.stop()
cam.close()
time.sleep(0.5)

focuser.waitingForFree()
time.sleep(0.5)

focuser.set(Focuser.OPT_MODE, 0)  # Disable motors
time.sleep(0.5)

# Reset chip registers as in original code
focuser.write(focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
time.sleep(0.5)

cv2.destroyAllWindows()

print("Shutdown complete.")
