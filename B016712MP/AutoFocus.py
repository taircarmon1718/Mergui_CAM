import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    # Physical focus range of the IMX477 PTZ lens
    MAX_FOCUS_VALUE = 1200

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.camera = camera
        self.debug = debug

        # Hailo incremental autofocus state
        self.stage = "idle"     # "idle" / "coarse" / "fine" / "done"
        self.step = 0
        self.threshold = 0
        self.max_dec = 0        # kept for compatibility, not used in new logic

        self.max_index = 0      # best focus position found so far
        self.max_value = -1.0   # best sharpness value
        self.last_value = -1.0
        self.dec_count = 0      # not used in new logic, kept for API compatibility
        self.focal = 0          # current focus position

        self.stage_end = 0      # last focus position for current stage

        # small rolling median (stabilizes values)
        self.buffer = []

    # ============================================================
    #  Tenengrad (more stable autofocus metric than Laplacian.var)
    # ============================================================
    def tenengrad(self, frame):
        """Compute Tenengrad sharpness measure for a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        fm = gx * gx + gy * gy
        return float(np.mean(fm))

    # Rolling 3-value median filter for stability
    def filt(self, v):
        self.buffer.append(v)
        if len(self.buffer) > 3:
            self.buffer.pop(0)
        return sorted(self.buffer)[len(self.buffer) // 2]

    # ============================================================
    # Initialize autofocus stage (coarse or fine)
    # ============================================================
    def _init_stage(self, step, start_pos, end_pos, stage):
        """
        Configure a scan stage.

        step      - focus increment per frame
        start_pos - starting focus value
        end_pos   - last focus value for this stage (inclusive)
        stage     - "coarse" or "fine"
        """
        if self.debug:
            print(f"[AF] init {stage}, start={start_pos}, end={end_pos}")

        self.stage = stage
        self.step = step

        # clamp to valid physical range
        start_pos = max(0, int(start_pos))
        end_pos = min(int(end_pos), self.MAX_FOCUS_VALUE)

        self.focal = start_pos
        self.stage_end = end_pos

        # reset best-so-far for this stage
        self.max_index = start_pos
        self.max_value = -1.0
        self.last_value = -1.0
        self.dec_count = 0
        self.buffer = []

        # move lens to starting position
        self.focuser.set(Focuser.OPT_FOCUS, start_pos)

    # ============================================================
    # Start autofocus (call once, before frames start arriving)
    # ============================================================
    def startFocus_hailo(self):
        """
        Start a two-stage autofocus:
        1) coarse scan: 0 → MAX in large steps
        2) fine scan: around the coarse maximum in small steps
        """
        # Coarse scan over the full physical range
        self._init_stage(
            step=80,              # coarse step size
            start_pos=0,
            end_pos=self.MAX_FOCUS_VALUE,
            stage="coarse"
        )

    # ============================================================
    # Incremental autofocus (call once per frame from app_callback)
    # ============================================================
    def stepFocus_hailo(self, frame):
        """
        Perform one autofocus step using the current frame.

        Returns:
            finished (bool): True when autofocus is complete.
            best_position (int or None): best focus position when finished.
        """
        # Nothing to do
        if self.stage in ("idle", "done"):
            return self.stage == "done", (
                self.max_index if self.stage == "done" else None
            )

        # Measure sharpness at the current focal position
        val = self.tenengrad(frame)
        val = self.filt(val)

        if self.debug:
            print(f"[AF] {self.stage} pos={self.focal} val={val:.2f}")

        # Update best focus position
        if val > self.max_value:
            self.max_value = val
            self.max_index = self.focal

        # --------------------------------------------------------
        # Decide whether the current stage is finished
        # (we simply scan the whole range for this stage)
        # --------------------------------------------------------
        if self.focal >= self.stage_end:
            # ---- COARSE STAGE FINISHED → switch to FINE ----
            if self.stage == "coarse":
                # Fine search window around the coarse maximum
                fine_start = max(0, self.max_index - 100)
                fine_end = min(self.max_index + 100, self.MAX_FOCUS_VALUE)

                if self.debug:
                    print(f"[AF] coarse done: peak={self.max_index}, "
                          f"fine_start={fine_start}, fine_end={fine_end}")

                self._init_stage(
                    step=10,              # fine step size
                    start_pos=fine_start,
                    end_pos=fine_end,
                    stage="fine"
                )
                # Not finished yet – fine scan still running
                return False, None

            # ---- FINE STAGE FINISHED → autofocus DONE ----
            if self.debug:
                print(f"[AF] FINISHED → best={self.max_index}, "
                      f"val={self.max_value:.2f}")

            # Move lens to the best focus position found
            self.focuser.set(Focuser.OPT_FOCUS, self.max_index)
            self.stage = "done"
            return True, self.max_index

        # --------------------------------------------------------
        # Continue scanning in this stage
        # --------------------------------------------------------
        self.focal += self.step
        if self.focal > self.stage_end:
            self.focal = self.stage_end

        self.focuser.set(Focuser.OPT_FOCUS, self.focal)

        return False, None
