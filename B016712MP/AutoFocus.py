'''
    Arducam programable zoom-lens autofocus component.

    Copyright (c) 2019-4 Arducam <http://www.arducam.com>.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
    OR OTHER DEALINGS IN THE SOFTWARE.
'''

import sys
import time
import math
import cv2  # sudo apt-get install python-opencv
import numpy as np

# from RpiCamera import Camera       # original demo
from B016712MP.RpiCamera import Camera  # adapt to your package
from B016712MP.Focuser import Focuser


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

        # ===== HAILO INCREMENTAL AF STATE (same logic as focusing/startFocus) ====
        self.h_stage = "idle"      # "idle" / "coarse" / "fine" / "done"
        self.h_step = 0
        self.h_threshold = 0.0
        self.h_max_dec_count = 0

        self.h_max_index = 0
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = 0
        # ========================================================================

    # ===============================
    # Original helper functions
    # ===============================
    def get_end_point(self):
        end_point = self.focuser.end_point[int(
            math.floor(self.focuser.get(Focuser.OPT_ZOOM) / 100.0)
        )]
        if self.debug:
            print("End Point: {}".format(end_point))
        return end_point

    def get_starting_point(self):
        starting_point = self.focuser.starting_point[int(
            math.ceil(self.focuser.get(Focuser.OPT_ZOOM) / 100.0)
        )]
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
        # width = image.shape[1]
        # height = image.shape[0]
        # image = image[(height / 4):((height / 4) * 3),
        #               (width / 4):((width / 4) * 3)]
        # return self.laplacian(image)
        # return self.sobel(image)
        return self.laplacian2(image)

    # ===============================
    # Original focusing logic (blocking)
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
            val = self.calculation(self.camera)
            val = self.filter(val)

            if self.debug:
                print("filter value = %d,focal_distance = %d" %
                      (val, focal_distance))

            if val > max_value:
                max_index = focal_distance
                max_value = val

            if last_value - val > threshold:
                if self.debug:
                    print("dec-----last_value = %lf,current_value = %lf" %
                          (last_value, val))
                dec_count += 1
            elif last_value - val != 0:
                dec_count = 0

            if dec_count > max_dec_count:
                break

            last_value = val
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
        last_time = time_list[0]
        if self.debug:
            print("total images = %d" % (len(images)))
            print("total time = %d" % (total_time))
        for i in range(len(images)):
            image = images.pop(0)
            width = image.shape[1]
            height = image.shape[0]
            # image = image[(height / 4):((height / 4) * 3),
            #               (width / 4):((width / 4) * 3)]
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
            int(((max_time - 0.0) / total_time)
                * (self.MAX_FOCUS_VALUE - starting_point)) + starting_point,
        )

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
            int(((max_time - 0.0) / total_time)
                * (self.MAX_FOCUS_VALUE - starting_point)) + starting_point,
        )

        if self.debug:
            print("focusing time = %lf" % (time.time() - begin))
        return max_index

    # =====================================================================
    # HAILO INCREMENTAL MODE (for app_callback)
    #  - SAME hill-climb logic as focusing(), just split across frames
    # =====================================================================
    def _hailo_init_stage(self, step, threshold, max_dec_count, start_pos, stage):
        if self.debug:
            print(f"[AF-H] start stage={stage}, step={step}, start_pos={start_pos}")

        self.value_buffer = []
        self.h_stage = stage
        self.h_step = int(step)
        self.h_threshold = float(threshold)
        self.h_max_dec_count = int(max_dec_count)

        self.h_max_index = int(start_pos)
        self.h_max_value = 0.0
        self.h_last_value = -1.0
        self.h_dec_count = 0
        self.h_focal_distance = int(start_pos)

        self.focuser.set(Focuser.OPT_FOCUS, self.h_focal_distance)

    def startFocus_hailo(self):
        """Initialize incremental autofocus (same params as startFocus)."""
        self.MAX_FOCUS_VALUE = self.get_end_point()
        starting_point = self.get_starting_point()
        if self.debug:
            print(
                f"[AF-H] init Hailo AF. MAX={self.MAX_FOCUS_VALUE}, "
                f"start={starting_point}"
            )

        self._hailo_init_stage(
            step=self.coarse_step_size,  # 100
            threshold=1,
            max_dec_count=2,
            start_pos=starting_point,
            stage="coarse",
        )

    def stepFocus_hailo(self, frame):
        """
        Perform ONE focusing step using the given RGB frame
        from Hailo (same logic as focusing()).

        Returns:
            finished (bool), best_pos (int or None)
        """
        if self.h_stage in ("idle", "done"):
            return (self.h_stage == "done"), (
                self.h_max_index if self.h_stage == "done" else None
            )

        # === one iteration of focusing() loop ===
        val = self.laplacian2(frame)
        val = self.filter(val)

        if self.debug:
            print(
                f"[AF-H] stage={self.h_stage}, pos={self.h_focal_distance}, "
                f"val={val:.2f}"
            )

        if val > self.h_max_value:
            self.h_max_value = val
            self.h_max_index = self.h_focal_distance

        if self.h_last_value >= 0:
            diff = self.h_last_value - val
            if diff > self.h_threshold:
                self.h_dec_count += 1
            elif diff != 0:
                self.h_dec_count = 0

        self.h_last_value = val

        if (
            self.h_dec_count > self.h_max_dec_count
            or self.h_focal_distance > self.MAX_FOCUS_VALUE
        ):
            if self.h_stage == "coarse":
                coarse_best = self.h_max_index - self.coarse_step_size
                if coarse_best < 0:
                    coarse_best = 0
                if self.debug:
                    print(
                        f"[AF-H] coarse done, best={self.h_max_index}, "
                        f"fine_start={coarse_best}"
                    )

                self._hailo_init_stage(
                    step=5,
                    threshold=1,
                    max_dec_count=3,
                    start_pos=coarse_best,
                    stage="fine",
                )
                return False, None
            else:
                if self.debug:
                    print(
                        f"[AF-H] DONE. Best focus={self.h_max_index}, "
                        f"val={self.h_max_value:.2f}"
                    )
                self.focuser.set(Focuser.OPT_FOCUS, int(self.h_max_index))
                self.h_stage = "done"
                return True, self.h_max_index

        self.h_focal_distance += self.h_step
        if self.h_focal_distance <= self.MAX_FOCUS_VALUE:
            self.focuser.set(Focuser.OPT_FOCUS, int(self.h_focal_distance))

        return False, None


if __name__ == "__main__":
    camera = Camera()
    camera.start_preview()
    focuser = Focuser(1)
    autoFocus = AutoFocus(focuser, camera)
    autoFocus.debug = True
    autoFocus.startFocus2()
    time.sleep(5)
    camera.stop_preview()
    camera.close()
