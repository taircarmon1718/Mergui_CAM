'''
    Arducam programable zoom-lens autofocus component.
    (original code, extended with Hailo incremental mode)
'''

import sys
import time
import math
import cv2  # sudo apt-get install python-opencv
import numpy as np
from RpiCamera import *          # kept for compatibility with original script
from Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1100
    value_buffer = []
    focuser = None
    camera = None
    debug = False
    coarse_step_size = 100

    def __init__(self, focuser, camera):
        self.focuser = focuser
        self.camera = camera

        # ------------------------------------------------------
        # Hailo incremental autofocus state (new)
        # ------------------------------------------------------
        self.h_stage = "idle"        # "idle" / "coarse" / "fine" / "done"
        self.h_step = 0
        self.h_threshold = 0
        self.h_max_dec_count = 0

        self.h_max_index = 0
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = 0

        # how many frames לחכות אחרי תזוזת פוקוס, לפני מדידה
        self.h_wait_frames = 0

    # ============================================================
    # ORIGINAL HELPERS
    # ============================================================
    def get_end_point(self):
        end_point = self.focuser.end_point[
            int(math.floor(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))
        ]
        if self.debug:
            print("End Point: {}".format(end_point))
        return end_point

    def get_starting_point(self):
        starting_point = self.focuser.starting_point[
            int(math.ceil(self.focuser.get(Focuser.OPT_ZOOM) / 100.0))
        ]
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

    def sobel(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_sobel = cv2.Sobel(img_gray, cv2.CV_16U, 1, 1)
        return cv2.mean(img_sobel)[0]

    def laplacian(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_sobel = cv2.Laplacian(img_gray, cv2.CV_16U)
        return cv2.mean(img_sobel)[0]

    def laplacian2(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_sobel = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return img_sobel

    def calculation(self, camera):
        image = camera.getFrame()
        # cropping was commented in original – נשאיר ככה
        # width = image.shape[1]
        # height = image.shape[0]
        # image = image[(height / 4):((height / 4) * 3),
        #               (width / 4):((width / 4) * 3)]
        # return self.laplacian(image)
        # return self.sobel(image)
        return self.laplacian2(image)

    # ============================================================
    # ORIGINAL BLOCKING FOCUSING (UNCHANGED)
    # ============================================================
    def focusing(self, step, threshold, max_dec_count):
        self.value_buffer = []
        max_index = self.focuser.get(Focuser.OPT_FOCUS)
        max_value = 0.0
        last_value = -1
        dec_count = 0
        focal_distance = max_index
        self.focuser.set(Focuser.OPT_FOCUS, focal_distance)

        while True:
            # Adjust focus
            self.focuser.set(Focuser.OPT_FOCUS, focal_distance)
            # Take image and calculate image clarity
            val = self.calculation(self.camera)
            val = self.filter(val)

            if self.debug:
                print("filter value = %d,focal_distance = %d" %
                      (val, focal_distance))

            # Find the maximum image clarity
            if val > max_value:
                max_index = focal_distance
                max_value = val

            # If the image clarity starts to decrease
            if last_value - val > threshold:
                if self.debug:
                    print("dec-----last_value = %lf,current_value = %lf" %
                          (last_value, val))
                dec_count += 1
            elif last_value - val != 0:
                dec_count = 0

            # Image clarity is reduced by several consecutive frames
            if dec_count > max_dec_count:
                break
            last_value = val

            # Increase the focal distance
            focal_distance = self.focuser.get(Focuser.OPT_FOCUS)
            focal_distance += step
            if focal_distance > self.MAX_FOCUS_VALUE:
                break

        return max_index, max_value

    def CoarseAdjustment(self, st_point, ed_point):
        images = []
        index_list = []
        eval_list = []
        time_list = []
        self.focuser.set(Focuser.OPT_FOCUS, st_point)

        image = self.camera.getFrame()
        time_list.append(time.time())
        images.append(image)

        self.focuser.set(Focuser.OPT_FOCUS, ed_point, 0)
        while self.focuser.isBusy():
            image = self.camera.getFrame()
            time_list.append(time.time())
            images.append(image)

        total_time = time_list[len(time_list) - 1] - time_list[0]
        index_list = np.arange(len(images))
        if self.debug:
            print("total images = %d" % (len(images)))
            print("total time = %d" % (total_time))

        for _ in range(len(images)):
            image = images.pop(0)
            result = self.laplacian2(image)
            eval_list.append(result)

        return eval_list, index_list, time_list

    def startFocus(self):
        begin = time.time()
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.MAX_FOCUS_VALUE = self.get_end_point()
        self.focuser.set(Focuser.OPT_FOCUS, self.get_starting_point())
        if self.debug:
            print("init time = %lf" % (time.time() - begin))

        begin = time.time()
        max_index, max_value = self.focusing(self.coarse_step_size, 1, 2)
        max_index = max_index - self.coarse_step_size
        if max_index < 0:
            max_index = 0
        self.focuser.set(Focuser.OPT_FOCUS, max_index)

        # Careful adjustment
        max_index, max_value = self.focusing(5, 1, 3)
        self.focuser.set(Focuser.OPT_FOCUS, max_index)

        if self.debug:
            print("focusing time = %lf" % (time.time() - begin))
        return max_index, max_value

    def startFocus2(self):
        begin = time.time()
        self.focuser.reset(Focuser.OPT_FOCUS)
        self.MAX_FOCUS_VALUE = self.get_end_point()
        starting_point = self.get_starting_point()

        if self.debug:
            print("init time = %lf" % (time.time() - begin))
        begin = time.time()
        eval_list, index_list, time_list = self.CoarseAdjustment(
            starting_point, self.MAX_FOCUS_VALUE
        )

        max_index = np.argmax(eval_list)
        total_time = time_list[len(time_list) - 1] - time_list[0]
        max_time = time_list[max_index - 1] - time_list[0]
        self.focuser.set(
            Focuser.OPT_FOCUS,
            int(((max_time - 0.0) / total_time) *
                (self.MAX_FOCUS_VALUE - starting_point)) + starting_point,
        )

        # Careful adjustment
        max_index, max_value = self.focusing(20, 1, 4)
        self.focuser.set(Focuser.OPT_FOCUS, max_index - 30)
        if self.debug:
            print("focusing time = %lf" % (time.time() - begin))
        return max_index, max_value

    def auxiliaryFocusing(self):
        begin = time.time()
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.MAX_FOCUS_VALUE = 20000
        starting_point = 0
        if self.debug:
            print("init time = %lf" % (time.time() - begin))
        begin = time.time()

        eval_list, index_list, time_list = self.CoarseAdjustment(
            starting_point, self.MAX_FOCUS_VALUE
        )

        max_index = np.argmax(eval_list)
        total_time = time_list[len(time_list) - 1] - time_list[0]
        max_time = time_list[max_index] - time_list[0]
        self.focuser.set(
            Focuser.OPT_FOCUS,
            int(((max_time - 0.0) / total_time) *
                (self.MAX_FOCUS_VALUE - starting_point)) + starting_point,
        )

        if self.debug:
            print("focusing time = %lf" % (time.time() - begin))
        return max_index

    # ============================================================
    # NEW: INCREMENTAL HAILO AUTOFOCUS
    # ============================================================
    def _hailo_init(self, step, threshold, max_dec_count, start_pos, stage):
        """
        Initialize one focusing stage (coarse or fine) for Hailo mode.
        """
        if self.debug:
            print(f"[AF-H] init stage={stage}, start={start_pos}")

        self.h_stage = stage
        self.h_step = step
        self.h_threshold = threshold
        self.h_max_dec_count = max_dec_count

        self.value_buffer = []          # reuse same buffer
        self.h_max_index = start_pos
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = start_pos

        self.h_wait_frames = 3          # wait a few frames after each jump
        self.focuser.set(Focuser.OPT_FOCUS, start_pos)

    def startFocus_hailo(self):
        """
        Prepare autofocus to run incrementally inside Hailo app_callback.
        Equivalent ללוגיקה של startFocus המקורי.
        """
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.MAX_FOCUS_VALUE = self.get_end_point()
        start = self.get_starting_point()

        self._hailo_init(
            step=self.coarse_step_size,   # 100
            threshold=1,
            max_dec_count=2,
            start_pos=start,
            stage="coarse",
        )

    def stepFocus_hailo(self, frame):
        """
        Perform one autofocus step using the given Hailo frame.

        Return:
            finished (bool), best_focus_position (int or None)
        """
        if self.h_stage in ("idle", "done"):
            return self.h_stage == "done", (
                self.h_max_index if self.h_stage == "done" else None
            )

        # wait a few frames after each focus move so the lens can settle
        if self.h_wait_frames > 0:
            self.h_wait_frames -= 1
            return False, None

        # --- calculate sharpness exactly as in original focusing() ---
        val = self.laplacian2(frame)
        val = self.filter(val)

        if self.debug:
            print(
                f"[AF] {self.h_stage} pos={self.h_focal_distance} "
                f"val={val:.2f}"
            )

        # update maximum
        if val > self.h_max_value:
            self.h_max_value = val
            self.h_max_index = self.h_focal_distance

        # decrease detection (same logic as original)
        if self.h_last_value - val > self.h_threshold:
            if self.debug:
                print(
                    "dec-----last_value = %lf,current_value = %lf"
                    % (self.h_last_value, val)
                )
            self.h_dec_count += 1
        elif self.h_last_value - val != 0:
            self.h_dec_count = 0

        # stop condition (same as original)
        if (
            self.h_dec_count > self.h_max_dec_count
            or self.h_focal_distance > self.MAX_FOCUS_VALUE
        ):
            if self.h_stage == "coarse":
                # move to fine stage around coarse best
                fine_start = self.h_max_index - self.coarse_step_size
                if fine_start < 0:
                    fine_start = 0
                self._hailo_init(
                    step=5,
                    threshold=1,
                    max_dec_count=3,
                    start_pos=fine_start,
                    stage="fine",
                )
                return False, None

            # fine stage done → apply best focus
            if self.debug:
                print(
                    "[AF] FINISHED → best=%d, val=%lf"
                    % (self.h_max_index, self.h_max_value)
                )
            self.focuser.set(Focuser.OPT_FOCUS, self.h_max_index)
            self.h_stage = "done"
            return True, self.h_max_index

        # continue scanning
        self.h_last_value = val
        self.h_focal_distance = self.focuser.get(Focuser.OPT_FOCUS)
        self.h_focal_distance += self.h_step
        if self.h_focal_distance > self.MAX_FOCUS_VALUE:
            self.h_focal_distance = self.MAX_FOCUS_VALUE + 1

        self.focuser.set(Focuser.OPT_FOCUS, self.h_focal_distance)
        self.h_wait_frames = 2
        return False, None


if __name__ == "__main__":
    # original standalone test (unchanged)
    camera = Camera()
    camera.start_preview()
    focuser = Focuser(1)
    autoFocus = AutoFocus(focuser, camera)
    autoFocus.debug = True
    autoFocus.startFocus2()
    time.sleep(5)
    camera.stop_preview()
    camera.close()
