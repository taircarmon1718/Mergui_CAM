import time
import math
import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    """
    Hailo-compatible autofocus controller.

    - Keeps the original Arducam focusing logic (Laplacian + hill-climb).
    - Does NOT own a camera.
    - You call:
        af.start_focus()           # once, to begin
        finished, best_pos = af.step(frame)  # every frame with a numpy RGB frame
    - When finished=True, best_pos is the chosen focus position.
    """

    def __init__(self, focuser: Focuser, debug: bool = False):
        self.focuser = focuser
        self.debug = debug

        self.MAX_FOCUS_VALUE = 1100
        self.coarse_step_size = 100

        # state for focusing
        self.value_buffer = []
        self.phase = "idle"        # "idle" / "coarse" / "fine" / "done"

        self.step = 0
        self.threshold = 0.0
        self.max_dec_count = 0

        self.max_index = 0
        self.max_value = 0.0
        self.last_value = -1.0
        self.dec_count = 0
        self.focal_distance = 0

    # ===============================
    # Utility functions
    # ===============================
    def get_end_point(self):
        end_point = self.focuser.end_point[
            int(math.floor(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))
        ]
        if self.debug:
            print(f"[AF] End Point: {end_point}")
        return end_point

    def get_starting_point(self):
        starting_point = self.focuser.starting_point[
            int(math.ceil(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))
        ]
        if self.debug:
            print(f"[AF] Starting Point: {starting_point}")
        return starting_point

    def filter(self, value):
        # simple median filter over last 3 values (same as original logic)
        max_len = 3
        self.value_buffer.append(value)
        if len(self.value_buffer) == max_len:
            sort_list = sorted(self.value_buffer)
            self.value_buffer.pop(0)
            return sort_list[math.ceil(max_len / 2)]
        return value

    # ===============================
    # Focus evaluation methods
    # ===============================
    def laplacian2(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # ===============================
    # Internal helpers
    # ===============================
    def _start_focusing_stage(self, step, threshold, max_dec_count, start_pos):
        """Initialize one focusing stage (coarse or fine)."""
        if self.debug:
            print(f"[AF] Start focusing stage: step={step}, start_pos={start_pos}")

        self.value_buffer = []
        self.step = step
        self.threshold = float(threshold)
        self.max_dec_count = max_dec_count

        self.max_index = int(start_pos)
        self.max_value = 0.0
        self.last_value = -1.0
        self.dec_count = 0
        self.focal_distance = int(start_pos)

        self.focuser.set(Focuser.OPT_FOCUS, self.focal_distance)

    # ===============================
    # Public API
    # ===============================
    def start_focus(self):
        """
        Start autofocus with coarse + fine search (non-blocking).
        Call this once, then repeatedly call step(frame).
        """
        begin = time.time()
        self.MAX_FOCUS_VALUE = self.get_end_point()
        starting_point = self.get_starting_point()
        if self.debug:
            print(f"[AF] init time = {time.time() - begin:.3f}s")
            print(f"[AF] MAX_FOCUS_VALUE={self.MAX_FOCUS_VALUE}, start={starting_point}")

        self.phase = "coarse"
        self._start_focusing_stage(
            step=self.coarse_step_size,
            threshold=1.0,
            max_dec_count=2,
            start_pos=starting_point,
        )

    def startFocus(self):
        """Alias for compatibility with old name."""
        self.start_focus()

    def startFocus2(self):
        """
        Hailo-friendly version of startFocus2:
        For compatibility, we just call start_focus().
        The time-based CoarseAdjustment from the original code is not
        safe inside the Hailo pipeline, so we keep the same hill-climb
        logic but without blocking.
        """
        self.start_focus()

    def step(self, frame):
        """
        Perform ONE autofocus iteration using the given RGB frame.

        Returns:
            finished (bool): True when autofocus is done.
            best_pos (int or None): Focus position if finished, else None.
        """
        if self.phase == "idle":
            return False, None
        if self.phase == "done":
            return True, self.max_index

        # compute sharpness
        val = self.laplacian2(frame)
        val = self.filter(val)

        if self.debug:
            print(
                f"[AF] phase={self.phase}, "
                f"focal_distance={self.focal_distance}, val={val:.2f}"
            )

        # update best
        if val > self.max_value:
            self.max_value = val
            self.max_index = self.focal_distance

        # update dec_count like original focusing()
        if self.last_value >= 0:
            diff = self.last_value - val
            if diff > self.threshold:
                self.dec_count += 1
            elif diff != 0:
                self.dec_count = 0

        self.last_value = val

        # termination for this stage
        if self.dec_count > self.max_dec_count or self.focal_distance > self.MAX_FOCUS_VALUE:
            if self.phase == "coarse":
                # start fine stage around best coarse position
                fine_start = max(0, self.max_index - self.coarse_step_size)
                if self.debug:
                    print(
                        f"[AF] Coarse done, best={self.max_index}, "
                        f"val={self.max_value:.2f}. Starting fine search..."
                    )
                self.phase = "fine"
                self._start_focusing_stage(
                    step=5,
                    threshold=1.0,
                    max_dec_count=3,
                    start_pos=fine_start,
                )
                return False, None
            else:
                # fine stage done → set lens to best and finish
                if self.debug:
                    print(
                        f"[AF] Fine done. Best focus={self.max_index}, "
                        f"val={self.max_value:.2f}"
                    )
                self.focuser.set(Focuser.OPT_FOCUS, int(self.max_index))
                self.phase = "done"
                return True, self.max_index

        # otherwise, continue this stage: move focus one step
        self.focal_distance += self.step
        if self.focal_distance <= self.MAX_FOCUS_VALUE:
            self.focuser.set(Focuser.OPT_FOCUS, int(self.focal_distance))

        return False, None
