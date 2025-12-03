import time
import math
import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1100
    value_buffer = []
    focuser = None
    camera = None
    debug = False
    coarse_step_size = 100

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.camera = camera
        self.debug = debug

        # --------- Hailo incremental AF state ---------
        self.h_stage = "idle"        # "idle" / "coarse" / "fine" / "done"
        self.h_step = 0
        self.h_threshold = 0.0
        self.h_max_dec = 0

        self.h_max_index = 0
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = 0
        # ---------------------------------------------


    # ===============================
    # Unified frame getter for original camera mode
    # (NOT used for Hailo)
    # ===============================
    def get_frame(self):
        """Return RGB image frame regardless of camera type."""
        if self.camera is None:
            raise RuntimeError("AutoFocus camera is None (Hailo mode uses process_frame_hailo(frame))")

        if hasattr(self.camera, "getFrame"):
            frame = self.camera.getFrame()
        elif hasattr(self.camera, "capture_array"):
            frame = self.camera.capture_array()
        else:
            raise AttributeError("Camera object has no supported frame capture method")

        if frame is None:
            raise ValueError("No frame captured from camera")

        # Convert RGBA → RGB if necessary
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return frame

    # ===============================
    # Utility functions
    # ===============================
    def get_end_point(self):
        end_point = self.focuser.end_point[int(math.floor(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))]
        if self.debug:
            print("End Point: {}".format(end_point))
        return end_point

    def get_starting_point(self):
        starting_point = self.focuser.starting_point[int(math.ceil(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))]
        if self.debug:
            print("Starting Point: {}".format(starting_point))
        return starting_point

    def filter(self, value):
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

    def calculation(self):
        image = self.get_frame()
        return self.laplacian2(image)

    # ===============================
    # Main focusing logic (original, blocking)
    # ===============================
    def focusing(self, step, threshold, max_dec_count):
        self.value_buffer = []
        max_index = self.focuser.get(Focuser.OPT_FOCUS)
        max_value = 0.0
        last_value = -1
        dec_count = 0
        focal_distance = max_index

        self.focuser.set(Focuser.OPT_FOCUS, focal_distance)
        while True:
            self.focuser.set(Focuser.OPT_FOCUS, focal_distance)
            val = self.calculation()
            val = self.filter(val)

            if self.debug:
                print(f"filter value = {val:.2f}, focal_distance = {focal_distance}")

            if val > max_value:
                max_index = focal_distance
                max_value = val

            if last_value - val > threshold:
                dec_count += 1
            elif last_value - val != 0:
                dec_count = 0

            if dec_count > max_dec_count:
                break

            last_value = val
            focal_distance += step
            if focal_distance > self.MAX_FOCUS_VALUE:
                break

        return max_index, max_value

    def CoarseAdjustment(self, st_point, ed_point):
        images, eval_list, time_list = [], [], []
        self.focuser.set(Focuser.OPT_FOCUS, st_point)

        image = self.get_frame()
        time_list.append(time.time())
        images.append(image)

        self.focuser.set(Focuser.OPT_FOCUS, ed_point, 0)
        while self.focuser.isBusy():
            image = self.get_frame()
            time_list.append(time.time())
            images.append(image)

        total_time = time_list[-1] - time_list[0]
        if self.debug:
            print(f"Total images = {len(images)}, total time = {total_time:.2f}s")

        for img in images:
            eval_list.append(self.laplacian2(img))

        index_list = np.arange(len(eval_list))
        return eval_list, index_list, time_list

    # ===============================
    # Public autofocus routines (original, blocking)
    # ===============================
    def startFocus(self):
        begin = time.time()
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.MAX_FOCUS_VALUE = self.get_end_point()
        self.focuser.set(Focuser.OPT_FOCUS, self.get_starting_point())
        if self.debug:
            print(f"init time = {time.time() - begin:.3f}s")

        begin = time.time()
        max_index, max_value = self.focusing(self.coarse_step_size, 1, 2)
        max_index = max(0, max_index - self.coarse_step_size)
        self.focuser.set(Focuser.OPT_FOCUS, max_index)
        max_index, max_value = self.focusing(5, 1, 3)
        self.focuser.set(Focuser.OPT_FOCUS, max_index)

        if self.debug:
            print(f"focusing time = {time.time() - begin:.3f}s")
        return max_index, max_value

    def startFocus2(self):
        begin = time.time()
        self.focuser.reset(Focuser.OPT_FOCUS)
        self.MAX_FOCUS_VALUE = self.get_end_point()
        starting_point = self.get_starting_point()
        if self.debug:
            print(f"init time = {time.time() - begin:.3f}s")

        eval_list, index_list, time_list = self.CoarseAdjustment(starting_point, self.MAX_FOCUS_VALUE)
        max_index = np.argmax(eval_list)
        total_time = time_list[-1] - time_list[0]
        max_time = time_list[max_index - 1] - time_list[0]

        target_focus = int(((max_time / total_time) * (self.MAX_FOCUS_VALUE - starting_point)) + starting_point)
        self.focuser.set(Focuser.OPT_FOCUS, target_focus)
        max_index, max_value = self.focusing(20, 1, 4)
        self.focuser.set(Focuser.OPT_FOCUS, max_index - 30)

        if self.debug:
            print(f"focusing time = {time.time() - begin:.3f}s")
        return max_index, max_value

    # ======================================================
    #       HAILO NON-BLOCKING AUTOFOCUS (NEW)
    #       runs safely from app_callback
    # ======================================================
    def _hailo_init_stage(self, step, threshold, max_dec_count, start_pos, stage_name):
        """Initialize one autofocus stage for Hailo (coarse or fine)."""
        if self.debug:
            print(f"[AF-H] Start {stage_name} stage: step={step}, start_pos={start_pos}")

        self.value_buffer = []
        self.h_stage = stage_name
        self.h_step = int(step)
        self.h_threshold = float(threshold)
        self.h_max_dec = int(max_dec_count)

        self.h_max_index = int(start_pos)
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = int(start_pos)

        self.focuser.set(Focuser.OPT_FOCUS, self.h_focal_distance)

    def startFocus_hailo(self):
        """
        Initialize autofocus for Hailo mode (non-blocking).
        Call this ONCE (e.g., after pan/tilt), then repeatedly call
        process_frame_hailo(frame) from app_callback.
        """
        self.MAX_FOCUS_VALUE = self.get_end_point()
        starting_point = self.get_starting_point()

        if self.debug:
            print(f"[AF-H] Hailo autofocus init. MAX={self.MAX_FOCUS_VALUE}, start={starting_point}")

        # Start with coarse search
        self._hailo_init_stage(
            step=self.coarse_step_size,
            threshold=1.0,
            max_dec_count=2,
            start_pos=starting_point,
            stage_name="coarse",
        )

    def process_frame_hailo(self, frame):
        """
        Run ONE autofocus step using the given RGB frame from Hailo.

        Returns:
            finished (bool): True when autofocus is done.
            best_pos (int or None): focus position if finished, else None.
        """
        if self.h_stage in ("idle", "done"):
            return (self.h_stage == "done"), (self.h_max_index if self.h_stage == "done" else None)

        # Evaluate sharpness
        val = self.laplacian2(frame)
        val = self.filter(val)

        if self.debug:
            print(
                f"[AF-H] stage={self.h_stage}, "
                f"pos={self.h_focal_distance}, val={val:.2f}"
            )

        # Track best
        if val > self.h_max_value:
            self.h_max_value = val
            self.h_max_index = self.h_focal_distance

        # Decrease tracking like original code
        if self.h_last_value >= 0:
            diff = self.h_last_value - val
            if diff > self.h_threshold:
                self.h_dec_count += 1
            elif diff != 0:
                self.h_dec_count = 0

        self.h_last_value = val

        # Termination condition for this stage
        if self.h_dec_count > self.h_max_dec or self.h_focal_distance > self.MAX_FOCUS_VALUE:
            if self.h_stage == "coarse":
                # Move to fine stage around best coarse value
                fine_start = max(0, self.h_max_index - self.coarse_step_size)
                fine_start = min(fine_start, self.MAX_FOCUS_VALUE)
                self._hailo_init_stage(
                    step=5,
                    threshold=1.0,
                    max_dec_count=3,
                    start_pos=fine_start,
                    stage_name="fine",
                )
                return False, None
            else:
                # Fine stage done → set best focus and finish
                if self.debug:
                    print(f"[AF-H] DONE. Best focus={self.h_max_index}, val={self.h_max_value:.2f}")
                self.focuser.set(Focuser.OPT_FOCUS, int(self.h_max_index))
                self.h_stage = "done"
                return True, self.h_max_index

        # Continue this stage: move focus
        self.h_focal_distance += self.h_step
        if self.h_focal_distance <= self.MAX_FOCUS_VALUE:
            self.focuser.set(Focuser.OPT_FOCUS, int(self.h_focal_distance))

        return False, None
