import math
import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    """
    Non-blocking autofocus for Hailo pipeline.

    Usage in your code:
        af = AutoFocus(focuser, camera=None, debug=True)
        af.startFocus_hailo()
        ...
        finished, best_pos = af.stepFocus_hailo(frame_from_hailo)
    """

    # Not used as a hard limit anymore; real limits come from focuser tables.
    MAX_FOCUS_VALUE = 20000

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.camera = camera   # not used in Hailo mode, kept for compatibility
        self.debug = debug

        # Hailo incremental AF state
        self.stage = "idle"          # "idle" / "coarse" / "fine" / "done"
        self.step = 0
        self.focal = 0               # current focus motor position
        self.stage_end = 0           # last position to scan in this stage

        self.max_index = 0           # best focus found in this stage
        self.max_value = -1.0        # best sharpness value

        # small rolling window to stabilize measurements
        self.buffer = []

    # ------------------------------------------------------------------
    # Helpers taken from original Arducam AutoFocus
    # ------------------------------------------------------------------
    def get_end_point(self):
        """Get focus end-point according to current zoom (same as original)."""
        return self.focuser.end_point[
            int(math.floor(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))
        ]

    def get_starting_point(self):
        """Get focus starting-point according to zoom (same as original)."""
        return self.focuser.starting_point[
            int(math.ceil(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))
        ]

    # ------------------------------------------------------------------
    # Sharpness measure – identical to original `laplacian2`
    # ------------------------------------------------------------------
    def _laplacian2(self, img):
        """Laplacian variance — same metric as in original code."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _sharpness(self, frame):
        """
        Compute sharpness on the **center crop** of the frame.
        This avoids being dominated by noisy borders.
        """
        h, w = frame.shape[:2]
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4
        crop = frame[y0:y1, x0:x1]

        val = self._laplacian2(crop)

        # rolling median over last 3 values
        self.buffer.append(val)
        if len(self.buffer) > 3:
            self.buffer.pop(0)
        return sorted(self.buffer)[len(self.buffer) // 2]

    # ------------------------------------------------------------------
    # Internal: init scan stage
    # ------------------------------------------------------------------
    def _init_stage(self, step, start_pos, end_pos, stage):
        """
        Configure scan stage (coarse or fine).
        We will linearly sweep from start_pos to end_pos in this stage.
        """
        start_pos = int(start_pos)
        end_pos = int(end_pos)

        if self.debug:
            print(f"[AF] init {stage}, start={start_pos}, end={end_pos}")

        self.stage = stage
        self.step = int(step)
        self.focal = start_pos
        self.stage_end = end_pos

        self.max_index = start_pos
        self.max_value = -1.0
        self.buffer = []

        # Move lens to starting position
        self.focuser.set(Focuser.OPT_FOCUS, start_pos)

    # ------------------------------------------------------------------
    # Public: start autofocus (call once before using stepFocus_hailo)
    # ------------------------------------------------------------------
    def startFocus_hailo(self):
        """
        Start a two-stage autofocus:
        1) coarse scan: from starting_point to end_point with large steps
        2) fine scan: around the coarse maximum with small steps
        """
        end_point = self.get_end_point()
        start_point = self.get_starting_point()

        # coarse scan over [start_point, end_point]
        self._init_stage(
            step=100,               # coarse step size (like original)
            start_pos=start_point,
            end_pos=end_point,
            stage="coarse"
        )

    # ------------------------------------------------------------------
    # Public: one autofocus step (call every frame from app_callback)
    # ------------------------------------------------------------------
    def stepFocus_hailo(self, frame):
        """
        Run a single autofocus step on the given frame.

        Returns:
            finished (bool): True when autofocus is complete.
            best_position (int or None): best focus position if finished.
        """
        if self.stage in ("idle", "done"):
            return self.stage == "done", (
                self.max_index if self.stage == "done" else None
            )

        # 1) Measure sharpness at current focus position
        val = self._sharpness(frame)

        if self.debug:
            print(f"[AF] {self.stage} pos={self.focal} val={val:.2f}")

        # Update best focus if this is the sharpest so far
        if val > self.max_value:
            self.max_value = val
            self.max_index = self.focal

        # 2) Check if we finished scanning the range for this stage
        if self.focal >= self.stage_end:
            # ---------------- COARSE STAGE → FINE STAGE ----------------
            if self.stage == "coarse":
                # fine window: ±100 around coarse maximum
                start_point = self.get_starting_point()
                end_point = self.get_end_point()

                fine_start = max(start_point, self.max_index - 100)
                fine_end = min(end_point, self.max_index + 100)

                if self.debug:
                    print(
                        f"[AF] coarse done: peak={self.max_index}, "
                        f"fine_start={fine_start}, fine_end={fine_end}"
                    )

                self._init_stage(
                    step=10,          # fine step size (like original 5–20)
                    start_pos=fine_start,
                    end_pos=fine_end,
                    stage="fine"
                )
                return False, None

            # ---------------------- FINE STAGE DONE ---------------------
            if self.debug:
                print(
                    f"[AF] FINISHED → best={self.max_index}, "
                    f"val={self.max_value:.2f}"
                )

            # Move lens to the best focus position found
            self.focuser.set(Focuser.OPT_FOCUS, self.max_index)
            self.stage = "done"
            return True, self.max_index

        # 3) Continue scanning in this stage
        self.focal += self.step
        if self.focal > self.stage_end:
            self.focal = self.stage_end

        self.focuser.set(Focuser.OPT_FOCUS, self.focal)

        return False, None
