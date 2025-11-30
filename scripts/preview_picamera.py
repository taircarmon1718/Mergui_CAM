# python
import sys
import time
import threading
import cv2
import numpy as np
from picamera2 import Picamera2

# Add PTZ library path
sys.path.append("/home/tair/Desktop/Mergui_CAM")
try:
    from B016712MP.Focuser import Focuser
    from B016712MP.AutoFocus import AutoFocus
    has_focuser = True
except Exception:
    has_focuser = False

# Camera Setup
cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (360, 640), "format": "RGB888"}
))
cam.start()
time.sleep(2)

# Optional PTZ / Focuser Setup (non-fatal)
focuser = None
if has_focuser:
    try:
        focuser = Focuser(1)
        focuser.set(Focuser.OPT_MODE, 1)
        time.sleep(0.2)
        focuser.set(Focuser.OPT_IRCUT, 0)
        time.sleep(0.2)
    except Exception:
        focuser = None
        has_focuser = False

# Shared state for capture + zoom/pan
latest_frame = None
lock = threading.Lock()
zoom = 1.0
zoom_center = None  # (x, y) in pixel coords relative to frame
last_print = 0

# Local fallback pan/tilt values when no focuser is available
fake_pan = 0
fake_tilt = 0

def get_ptz_coords():
    # Return pan, tilt raw motor values (or fallback)
    if focuser is not None:
        try:
            pan = focuser.get(Focuser.OPT_MOTOR_X)
            tilt = focuser.get(Focuser.OPT_MOTOR_Y)
            return pan, tilt
        except Exception:
            return None, None
    else:
        return fake_pan, fake_tilt

def set_ptz(pan_val, tilt_val):
    global fake_pan, fake_tilt
    if focuser is not None:
        try:
            focuser.set(Focuser.OPT_MOTOR_X, int(pan_val))
            focuser.set(Focuser.OPT_MOTOR_Y, int(tilt_val))
        except Exception:
            # If hardware fails, fall back to local values for display
            fake_pan, fake_tilt = pan_val, tilt_val
    else:
        fake_pan, fake_tilt = pan_val, tilt_val

def apply_digital_zoom(frame, z, center):
    h, w = frame.shape[:2]
    if z <= 1.0:
        return frame
    # crop size
    crop_w = int(w / z)
    crop_h = int(h / z)
    cx, cy = center
    x1 = max(0, int(cx - crop_w // 2))
    y1 = max(0, int(cy - crop_h // 2))
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    # adjust if crop near border
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def capture_thread():
    global latest_frame, zoom_center
    while True:
        frame = cam.capture_array()
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        with lock:
            if zoom_center is None:
                zoom_center = (w // 2, h // 2)
            # apply digital zoom using current center and zoom
            fz = apply_digital_zoom(frame, zoom, zoom_center)
            latest_frame = fz

threading.Thread(target=capture_thread, daemon=True).start()

# Main Preview Loop
try:
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Overlay: central rectangle
        height, width, _ = frame.shape
        rect_w, rect_h = 200, 100
        top_left = ((width - rect_w) // 2, (height - rect_h) // 2)
        bottom_right = ((width + rect_w) // 2, (height + rect_h) // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Coordinates text
        coord_text = f"Rect: ({top_left[0]}, {top_left[1]})"
        cv2.putText(frame, coord_text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # PTZ / Zoom info
        pan, tilt = get_ptz_coords()
        pan_str = str(pan) if pan is not None else "N/A"
        tilt_str = str(tilt) if tilt is not None else "N/A"
        z_center = zoom_center if zoom_center is not None else (width//2, height//2)
        zoom_text = f"Zoom: {zoom:.2f}  Center: ({z_center[0]}, {z_center[1]})"
        servo_text = f"Pan: {pan_str}  Tilt: {tilt_str}"
        cv2.putText(frame, zoom_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, servo_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Print to console periodically
        now = time.time()
        if now - last_print > 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] {zoom_text} | {servo_text}")
            last_print = now

        cv2.imshow("Mergui PTZ Camera Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        # Controls:
        # q - quit, + / = - zoom in, - - zoom out
        # Arrow keys to move zoom center (handled via raw codes)
        # 1 - tilt 0, pan 0
        # 2 - tilt 90, pan 0
        # 3 - tilt 180, pan 0
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            with lock:
                zoom = min(8.0, zoom + 0.2)
        elif key == ord('-'):
            with lock:
                zoom = max(1.0, zoom - 0.2)
        elif key == 81:  # left arrow
            with lock:
                cx, cy = zoom_center
                zoom_center = (max(0, cx - 10), cy)
        elif key == 83:  # right arrow
            with lock:
                cx, cy = zoom_center
                zoom_center = (min(width, cx + 10), cy)
        elif key == 82:  # up arrow
            with lock:
                cx, cy = zoom_center
                zoom_center = (cx, max(0, cy - 10))
        elif key == 84:  # down arrow
            with lock:
                cx, cy = zoom_center
                zoom_center = (cx, min(height, cy + 10))
        elif key == ord('z'):  # reset zoom
            with lock:
                zoom = 1.0
                zoom_center = (width // 2, height // 2)
        elif key == ord('1'):
            # Move tilt to 0, pan to 0
            set_ptz(0, 0)
            last_print = 0  # force immediate print/update
            print("Moved to Pan=0 Tilt=0")
        elif key == ord('2'):
            set_ptz(0, 90)
            last_print = 0
            print("Moved to Pan=0 Tilt=90")
        elif key == ord('3'):
            set_ptz(0, 180)
            last_print = 0
            print("Moved to Pan=0 Tilt=180")

except KeyboardInterrupt:
    pass
finally:
    cam.stop()
    if focuser is not None:
        try:
            focuser.waitIdle()
            focuser.set(Focuser.OPT_MODE, 0)
            focuser.resetAll()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("Shutdown complete.")