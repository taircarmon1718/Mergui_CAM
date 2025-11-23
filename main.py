# live_cam_track_target.py
# ------------------------------------------------------------
# Live camera (or RTSP/file) YOLOv8 + ByteTrack tracking with interactive ID selection.
# - Click on a box to set the target ID to follow.
# - Press 'c' to clear target, 'q' to quit.
# - Displays live only (no saving).
#
# Requirements:
#   pip install ultralytics opencv-python numpy
# ------------------------------------------------------------

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

# ====================== USER CONFIG =========================
WEIGHTS      = r"downloaded_weights/artifact_yolo-horse-v1_v0/best.pt"   # path to your YOLO weights (.pt)
SOURCE       = "videos/gilli_comp.mp4"                              # "0" for default webcam, or RTSP URL, or path to video
CONF         = 0.45                             # higher conf reduces clutter
IMGSZ        = 640                              # smaller is faster; 640/736 good for live
TRACKER_YAML = "bytetrack.yaml"                 # shipped with ultralytics
ONLY_HORSE   = True                             # if "horse" is class id 0 in your model
DEVICE       = "cpu"                            # options: "cpu", 0 (CUDA), "mps" (Apple Silicon)
# ===========================================================

# --- Global state for mouse callback ---
clicked_point = None
target_id = None

def on_mouse(event, x, y, flags, userdata):
    """OpenCV mouse callback to pick a track by clicking inside its box."""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def pick_id_by_click(boxes_xyxy, ids, click_xy):
    """Return the track id of the box that contains click_xy, or None."""
    if click_xy is None or boxes_xyxy is None or ids is None:
        return None
    cx, cy = click_xy
    for (x1, y1, x2, y2), tid in zip(boxes_xyxy, ids):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return int(tid)
    return None

def draw_boxes(img, boxes_xyxy, ids, clss, confs, highlight_id=None):
    """Minimal drawing: highlight target; dim others."""
    if boxes_xyxy is None:
        return img

    img_out = img.copy()
    for (x1, y1, x2, y2), tid, c, p in zip(boxes_xyxy, ids, clss, confs):
        tid = int(tid) if tid is not None else -1
        label = f"ID {tid}"
        color = (0, 255, 0) if (highlight_id is not None and tid == highlight_id) else (200, 200, 200)
        thickness = 3 if (highlight_id is not None and tid == highlight_id) else 1

        cv2.rectangle(img_out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(
            img_out, label,
            (int(x1), int(max(y1-6, 10))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
    return img_out

def run_live():
    global clicked_point, target_id

    # Load model
    w = Path(WEIGHTS)
    if not w.exists():
        print(f"[ERROR] Weights not found: {w.resolve()}")
        return
    print(f"[INFO] Loading model: {w.resolve()}")
    model = YOLO(str(w))

    # Optional fuse (small speedup)
    try:
        model.fuse()
    except Exception:
        pass

    classes = [0] if ONLY_HORSE else None

    # Prepare window + mouse
    win_name = "Live Tracking (click a box to set target | 'c' clear | 'q' quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    # Stream results from Ultralytics tracker
    print(f"[INFO] Starting stream: source={SOURCE}, device={DEVICE}, imgsz={IMGSZ}")
    fps_t0 = time.time()
    fps_n = 0
    current_fps = 0.0

    try:
        for res in model.track(
            source=SOURCE,
            conf=CONF,
            imgsz=IMGSZ,
            tracker=TRACKER_YAML,
            persist=True,
            stream=True,       # IMPORTANT: gives us per-frame results
            show=False,        # we draw ourselves (needed for mouse picking)
            save=False,
            classes=classes,
            device=DEVICE,
            verbose=False,
        ):
            # Original frame
            frame = res.orig_img
            if frame is None:
                continue

            # Extract tracked boxes (+ IDs) if they exist
            boxes_xyxy, ids, clss, confs = None, None, None, None
            if res.boxes is not None:
                b = res.boxes
                # b.id may be None until tracker warms up; guard carefully
                ids   = b.id.cpu().numpy().astype(int) if (b.id is not None) else None
                xyxy  = b.xyxy.cpu().numpy().astype(np.float32) if (b.xyxy is not None) else None
                clss  = b.cls.cpu().numpy().astype(int) if (b.cls is not None) else None
                confs = b.conf.cpu().numpy().astype(float) if (b.conf is not None) else None
                boxes_xyxy = xyxy

            # Handle click -> pick target ID
            if clicked_point is not None and boxes_xyxy is not None and ids is not None:
                chosen = pick_id_by_click(boxes_xyxy, ids, clicked_point)
                if chosen is not None:
                    target_id = chosen
                    print(f"[INFO] Target set to ID {target_id}")
                clicked_point = None  # consume click

            # Draw
            canvas = draw_boxes(frame, boxes_xyxy, ids if ids is not None else [],
                                clss if clss is not None else [], confs if confs is not None else [],
                                highlight_id=target_id)

            # HUD: FPS + target status
            fps_n += 1
            if fps_n >= 15:
                now = time.time()
                dt = now - fps_t0
                current_fps = fps_n / dt if dt > 0 else 0.0
                fps_t0 = now
                fps_n = 0

            hud = f"FPS: {current_fps:.1f}"
            if target_id is None:
                hud += " | Target: (none) - click a box"
            else:
                # Check if target visible
                visible = False
                if ids is not None:
                    visible = any(int(t) == int(target_id) for t in ids)
                hud += f" | Target: ID {target_id} ({'visible' if visible else 'lost'})"
            cv2.putText(canvas, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2, cv2.LINE_AA)

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):   # quit
                break
            if key == ord('c'):   # clear target
                target_id = None

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("[OK] Stopped.")

if __name__ == "__main__":
    run_live()
