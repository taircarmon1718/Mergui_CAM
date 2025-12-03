import sys
import time
import math
import cv2
import numpy as np

from B016712MP.Focuser import Focuser


class AutoFocus:
    # ✔ FIXED — true physical range of the Arducam IMX477 PTZ lens
    MAX_FOCUS_VALUE = 1200

    value_buffer = []
    focuser = None
    camera = None
    debug = False
    coarse_step_size = 100   # original Arducam

    def __init__(self, focuser, camera=None):
        self.focuser = focuser
        self.camera = camera

        # incremental (Hailo) AF state:
        self.h_stage = "idle"     # idle / coarse / fine / done
        self.h_step = 0
        self.h_threshold = 0
        self.h_max_dec_count = 0

        self.h_max_index = 0
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = 0


    # ===============================
    # Frame sharpness methods
    # ===============================
    def laplacian2(self, img):
        """Laplacian variance = image sharpness score"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()


    def filter(self, value):
        """3-value rolling median filter"""
        max_len = 3
        self.value_buffer.append(value)
        if len(self.value_buffer) == max_len:
            sorted_vals = sorted(self.value_buffer)
            self.value_buffer.pop(0)
            return sorted_vals[max_len // 2]
        return value


    # ===============================
    # Original BLOCKING Arducam autofocus
    # (unchanged)
    # ===============================
    def focusing(self, step, threshold, max_dec_count):
        """Original Arducam hill-climb autofocus loop"""
        self.value_buffer = []

        max_index = self.focuser.get(Focuser.OPT_FOCUS)
        max_value = 0.0
        last_value = -1
        dec_count = 0

        focal = max_index
        self.focuser.set(Focuser.OPT_FOCUS, focal)

        while True:
            self.focuser.set(Focuser.OPT_FOCUS, focal)

            img = self.camera.getFrame()
            val = self.filter(self.laplacian2(img))

            if self.debug:
                print(f"[AF] pos={focal} val={val:.2f}")

            if val > max_value:
                max_value = val
                max_index = focal

            # detect decreases
            if last_value - val > threshold:
                dec_count += 1
            elif last_value - val != 0:
                dec_count = 0

            if dec_count > max_dec_count:
                break

            last_value = val
            focal += step

            if focal > self.MAX_FOCUS_VALUE:   # ✔ FIXED correct range
                break

        return max_index, max_value


    def startFocus(self):
        """Original Arducam startFocus()"""
        # reset to 0
        self.focuser.set(Focuser.OPT_FOCUS, 0)

        # start at 0
        start = 0

        # coarse pass
        coarse_best, _ = self.focusing(self.coarse_step_size, 1, 2)
        coarse_best = max(0, coarse_best - self.coarse_step_size)

        self.focuser.set(Focuser.OPT_FOCUS, coarse_best)

        # fine scan
        fine_best, fine_val = self.focusing(5, 1, 3)
        self.focuser.set(Focuser.OPT_FOCUS, fine_best)

        return fine_best, fine_val


    # ===============================
    # NON-BLOCKING autofocus for Hailo pipeline
    # (works frame-by-frame inside app_callback)
    # ===============================
    def _hailo_init(self, step, threshold, max_dec, start_pos, stage):
        if self.debug:
            print(f"[AF-H] init stage={stage} start={start_pos}")

        self.value_buffer = []
        self.h_stage = stage
        self.h_step = step
        self.h_threshold = threshold
        self.h_max_dec_count = max_dec

        self.h_max_index = start_pos
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = start_pos

        self.focuser.set(Focuser.OPT_FOCUS, start_pos)



    def startFocus_hailo(self):
        """Initialize incremental autofocus."""
        start_pos = 0   # IMX477 always starts focus range at 0

        self._hailo_init(
            step=self.coarse_step_size,   # 100
            threshold=1,
            max_dec=2,
            start_pos=start_pos,
            stage="coarse"
        )


    def stepFocus_hailo(self, frame):
        """
        Perform one autofocus step (called in app_callback)
        Returns:
            finished (bool), best_focus_position (int or None)
        """
        if self.h_stage in ("idle", "done"):
            return self.h_stage == "done", (
                self.h_max_index if self.h_stage == "done" else None
            )

        # --- compute sharpness on Hailo frame ---
        val = self.filter(self.laplacian2(frame))

        if self.debug:
            print(f"[AF-H] stage={self.h_stage}, pos={self.h_focal_distance}, val={val:.2f}")

        # update max
        if val > self.h_max_value:
            self.h_max_value = val
            self.h_max_index = self.h_focal_distance

        # detect decreases
        if self.h_last_value >= 0:
            if self.h_last_value - val > self.h_threshold:
                self.h_dec_count += 1
            elif self.h_last_value - val != 0:
                self.h_dec_count = 0

        self.h_last_value = val

        # ---- STOP CONDITION ----
        if (
            self.h_dec_count > self.h_max_dec_count or
            self.h_focal_distance > self.MAX_FOCUS_VALUE   # ✔ FIXED range
        ):
            if self.h_stage == "coarse":
                # prepare fine stage
                fine_start = max(0, self.h_max_index - self.coarse_step_size)

                if self.debug:
                    print(f"[AF-H] coarse done. peak={self.h_max_index}, fine_start={fine_start}")

                self._hailo_init(
                    step=5,
                    threshold=1,
                    max_dec=3,
                    start_pos=fine_start,
                    stage="fine"
                )
                return False, None

            # ✓ DONE — apply best focus
            if self.debug:
                print(f"[AF-H] DONE best={self.h_max_index} val={self.h_max_value:.2f}")

            self.focuser.set(Focuser.OPT_FOCUS, self.h_max_index)
            self.h_stage = "done"
            return True, self.h_max_index

        # ---- Continue scanning ----
        self.h_focal_distance += self.h_step

        if self.h_focal_distance <= self.MAX_FOCUS_VALUE:
            self.focuser.set(Focuser.OPT_FOCUS, self.h_focal_distance)

        return False, None
