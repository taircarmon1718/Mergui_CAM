import sys
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

    def __init__(self, focuser, camera):
        self.focuser = focuser
        self.camera = camera

    # ===============================
    # Unified frame getter for both RpiCamera and Picamera2
    # ===============================
    def get_frame(self):
        """Return RGB image frame regardless of camera type."""
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
    # Main focusing logic
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
    # Public autofocus routines
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
