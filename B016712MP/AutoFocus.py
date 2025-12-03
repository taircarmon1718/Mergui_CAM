import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1200  # True IMX477 PTZ physical range

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.camera = camera
        self.debug = debug

        # Hailo incremental autofocus state
        self.stage = "idle"     # idle / coarse / fine / done
        self.step = 0
        self.threshold = 0
        self.max_dec = 0

        self.max_index = 0
        self.max_value = 0
        self.last_value = -1
        self.dec_count = 0
        self.focal = 0

        # small rolling median (stabilizes values)
        self.buffer = []


    # ============================================================
    #  NEW: Tenengrad (stable autofocus metric)
    # ============================================================
    def tenengrad(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        fm = gx * gx + gy * gy
        return float(np.mean(fm))


    # rolling 3-value median
    def filt(self, v):
        self.buffer.append(v)
        if len(self.buffer) > 3:
            self.buffer.pop(0)
        return sorted(self.buffer)[len(self.buffer)//2]


    # ============================================================
    # Initialize autofocus stage
    # ============================================================
    def _init_stage(self, step, threshold, max_dec, start_pos, stage):
        if self.debug:
            print(f"[AF] init {stage}, start={start_pos}")

        self.stage = stage
        self.step = step
        self.threshold = threshold
        self.max_dec = max_dec

        self.max_index = start_pos
        self.max_value = 0
        self.last_value = -1
        self.dec_count = 0
        self.focal = start_pos
        self.buffer = []

        self.focuser.set(Focuser.OPT_FOCUS, start_pos)


    # ============================================================
    # Start autofocus (call once)
    # ============================================================
    def startFocus_hailo(self):
        self._init_stage(
            step=80,            # coarse step
            threshold=0.1,      # very stable metric → low sensitivity
            max_dec=2,
            start_pos=0,
            stage="coarse"
        )


    # ============================================================
    # Incremental autofocus (call every frame)
    # ============================================================
    def stepFocus_hailo(self, frame):
        """
        returns: (finished: bool, best_position: int or None)
        """
        if self.stage in ("idle", "done"):
            return self.stage == "done", (
                self.max_index if self.stage == "done" else None
            )

        # measure sharpness
        val = self.tenengrad(frame)
        val = self.filt(val)

        if self.debug:
            print(f"[AF] {self.stage} pos={self.focal} val={val:.2f}")

        # update max
        if val > self.max_value:
            self.max_value = val
            self.max_index = self.focal

        # detect decrease
        if self.last_value >= 0:
            if self.last_value - val > self.threshold:
                self.dec_count += 1
            elif self.last_value != val:
                self.dec_count = 0

        self.last_value = val

        # ====================================================
        # STOP CONDITION for current stage
        # ====================================================
        if self.dec_count > self.max_dec or self.focal >= self.MAX_FOCUS_VALUE:

            # ---------------- COARSE → FINE ----------------
            if self.stage == "coarse":
                fine_start = max(0, self.max_index - 100)

                if self.debug:
                    print(f"[AF] coarse done: peak={self.max_index}, fine_start={fine_start}")

                self._init_stage(
                    step=10,
                    threshold=0.05,
                    max_dec=4,
                    start_pos=fine_start,
                    stage="fine"
                )
                return False, None

            # ---------------------- DONE ---------------------
            if self.debug:
                print(f"[AF] FINISHED → best={self.max_index}, val={self.max_value:.2f}")

            self.focuser.set(Focuser.OPT_FOCUS, self.max_index)
            self.stage = "done"
            return True, self.max_index

        # ====================================================
        # Continue scanning
        # ====================================================
        self.focal += self.step

        if self.focal <= self.MAX_FOCUS_VALUE:
            self.focuser.set(Focuser.OPT_FOCUS, self.focal)

        return False, None
