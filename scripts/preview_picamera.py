# python
import sys
import time
import threading
import cv2
import numpy as np

# Ensure PTZ library path (adjust if your project is in a different place)
sys.path.append("/Users/taircarmon/Desktop/Mergui_CAM")

# Picamera2 import (runtime required)
try:
    from picamera2 import Picamera2
except Exception as e:
    raise RuntimeError("picamera2 is required to run this script") from e

# Force import of focuser classes; fail if not present (user requested forced setup)
try:
    from B016712MP.Focuser import Focuser
    from B016712MP.AutoFocus import AutoFocus
except Exception as e:
    raise RuntimeError("Focuser library not found at /Users/taircarmon/Desktop/Mergui_CAM") from e

# Camera Setup (do not change this configuration)
cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (360, 640), "format": "RGB888"}
))
cam.start()
time.sleep(2)

# Force PTZ / Focuser Setup (will raise on failure)
focuser = None
try:
    focuser = Focuser(1)
    # Enable motors
    focuser.set(Focuser.OPT_MODE, 1)
    time.sleep(0.5)

    print("Disabling IR-CUT...")
    focuser.set(Focuser.OPT_IRCUT, 0)
    time.sleep(0.5)

    # Optional small initial movements to engage motors
    print("Initial pan/tilt movement...")
    try:
        focuser.set(Focuser.OPT_MOTOR_X, 300)
        time.sleep(2)
        focuser.set(Focuser.OPT_MOTOR_Y, 25)
        time.sleep(2)
        # Reset pan to 0 (keep pan at 0 as requested)
        focuser.set(Focuser.OPT_MOTOR_X, 0)
        time.sleep(1)
    except Exception as e:
        raise RuntimeError("Initial PTZ movement failed") from e

    # Try running autofocus if available
    try:
        auto_focus = AutoFocus(focuser, cam)
        auto_focus.debug = True
        max_index, max_value = auto_focus.startFocus2()
        print(f"Autofocus completed: index={max_index}, value={max_value}")
        time.sleep(1)
    except Exception:
        # autofocus optional; continue if it fails
        pass

    print("PTZ initialization complete.")
except Exception as e:
    # Force failure so user must fix hardware/library before proceeding
    print("Focuser initialization failed. Aborting.")
    try:
        cam.stop()
        cam.close()
    except Exception:
        pass
    raise

# Shared state for capture + zoom/pan
latest_frame = None
lock = threading.Lock()
zoom = 1.0
zoom_center = None  # (x, y) in pixel coords relative to frame
last_print = 0

# Fallback local values (kept in sync if hardware calls fail)
fake_pan = 0
fake_tilt = 0

def get_ptz_coords():
    # Return pan, tilt raw motor values (or fallback)
    global fake_pan, fake_tilt
    try:
        pan = focuser.get(Focuser.OPT_MOTOR_X)
        tilt = focuser.get(Focuser.OPT_MOTOR_Y)
        return pan, tilt
    except Exception:
        return fake_pan, fake_tilt

def set_ptz(pan_val, tilt_val):
    global fake_pan, fake_tilt
    try:
        focuser.set(Focuser.OPT_MOTOR_X, int(pan_val))
        focuser.set(Focuser.OPT_MOTOR_Y, int(tilt_val))
    except Exception:
        # If hardware fails for any reason, keep local values for display
        fake_pan, fake_tilt = pan_val, tilt_val

def apply_digital_zoom(frame, z, center):
    """
    Returns (processed_frame, adjusted_center).
    adjusted_center is the actual center used for the crop (clamped).
    """
    h, w = frame.shape[:2]
    cx, cy = center if center is not None else (w // 2, h // 2)
    cx = int(min(max(cx, 0), w))
    cy = int(min(max(cy, 0), h))

    if z <= 1.0:
        return frame, (cx, cy)

    crop_w = int(w / z)
    crop_h = int(h / z)

    x1 = int(cx - crop_w // 2)
    y1 = int(cy - crop_h // 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    if x1 < 0:
        x1 = 0
        x2 = crop_w
    if y1 < 0:
        y1 = 0
        y2 = crop_h
    if x2 > w:
        x2 = w
        x1 = max(0, w - crop_w)
    if y2 > h:
        y2 = h
        y1 = max(0, h - crop_h)

    actual_w = x2 - x1
    actual_h = y2 - y1
    if actual_w <= 0 or actual_h <= 0:
        return frame, (w // 2, h // 2)

    cropped = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    adjusted_cx = x1 + actual_w // 2
    adjusted_cy = y1 + actual_h // 2

    return resized, (adjusted_cx, adjusted_cy)

def capture_thread():
    global latest_frame, zoom_center, zoom
    while True:
        frame = cam.capture_array()
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        with lock:
            if zoom_center is None:
                zoom_center = (w // 2, h // 2)
            fz, adjusted_center = apply_digital_zoom(frame, zoom, zoom_center)
            # Store the actual center used for crop so pan/arrow moves match what is displayed
            zoom_center = adjusted_center
            latest_frame = fz

# Start capture thread after PTZ init (PTZ was forced above)
threading.Thread(target=capture_thread, daemon=True).start()

# Main Preview Loop
try:
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        height, width, _ = frame.shape
        rect_w, rect_h = 200, 100
        top_left = ((width - rect_w) // 2, (height - rect_h) // 2)
        bottom_right = ((width + rect_w) // 2, (height + rect_h) // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        coord_text = f"Rect: ({top_left[0]}, {top_left[1]})"
        cv2.putText(frame, coord_text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        pan, tilt = get_ptz_coords()
        pan_str = str(pan) if pan is not None else "N/A"
        tilt_str = str(tilt) if tilt is not None else "N/A"
        z_center = zoom_center if zoom_center is not None else (width // 2, height // 2)
        zoom_text = f"Zoom: {zoom:.2f}  Center: ({z_center[0]}, {z_center[1]})"
        servo_text = f"Pan: {pan_str}  Tilt: {tilt_str}"
        cv2.putText(frame, zoom_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, servo_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        now = time.time()
        if now - last_print > 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] {zoom_text} | {servo_text}")
            last_print = now

        cv2.imshow("Mergui PTZ Camera Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        # Controls:
        # q - quit, + / = - zoom in, - - zoom out
        # Arrow keys to move zoom center (raw codes)
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
            set_ptz(0, 0)
            last_print = 0
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
    print("Stopping camera and resetting PTZ...")
    try:
        cam.stop()
        cam.close()
    except Exception:
        pass

    if focuser is not None:
        try:
            focuser.waitingForFree()
            time.sleep(0.5)
            focuser.set(Focuser.OPT_MODE, 0)  # Disable motors
            time.sleep(0.5)
            try:
                focuser.write(focuser.CHIP_I2C_ADDR, 0x11, 0x0001)
                time.sleep(0.5)
            except Exception:
                pass
        except Exception:
            pass

    cv2.destroyAllWindows()
    print("Shutdown complete.")