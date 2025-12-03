import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1200

    # Generous wait time for stabilization
    FRAMES_TO_WAIT = 8

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.debug = debug

        self.stage = "idle"
        self.best_pos = 0
        self.best_score = -1.0
        self.current_pos = 0
        self.wait_counter = 0

        # Scan range
        self.scan_start = 0
        self.scan_end = 0
        self.step_size = 0

    # =================================================================
    # The heart of the algorithm - Improved sharpness calculation
    # =================================================================
    def get_sharpness(self, frame):
        h, w = frame.shape[:2]

        # 1. Aggressive center crop (only the object matters)
        # Taking a small square in the middle (approx 1/4 of dimensions)
        roi = frame[h // 2 - h // 8: h // 2 + h // 8, w // 2 - w // 8: w // 2 + w // 8]

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # 3. Contrast Boosting (Histogram Equalization) - Critical!
        # This turns dull gray into strong black-and-white, highlighting lines.
        gray = cv2.equalizeHist(gray)

        # 4. Noise Filtering (Gaussian Blur)
        # This erases the "snow" so we don't focus on it.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 5. Edge detection with Sobel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 6. Compute Edge Magnitude
        mag = cv2.magnitude(gx, gy)

        # 7. Noise Threshold - The most important trick
        # We ignore any "line" that is too weak (below 50).
        # This ensures we measure only real edges and not background noise.
        _, mag = cv2.threshold(mag, 50, 255, cv2.THRESH_TOZERO)

        # Return the mean of strong edges
        return float(np.mean(mag))

    # =================================================================
    # Process Management
    # =================================================================
    def startFocus_hailo(self):
        print("[AF] Resetting lens to 0...")
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.stage = "reset_wait"
        self.wait_counter = 15

    def stepFocus_hailo(self, frame):
        # Debounce/Wait mechanism
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return False, None

        # -- Stage 1: Start --
        if self.stage == "reset_wait":
            print("[AF] Starting Scan...")
            self.stage = "scanning"
            self.current_pos = 0
            self.scan_end = self.MAX_FOCUS_VALUE
            # Smaller steps from the start to avoid missing real peaks
            self.step_size = 25
            self.best_score = -1
            self.best_pos = 0
            return False, None

        # -- Stage 2: Linear Scan --
        if self.stage == "scanning":
            val = self.get_sharpness(frame)

            # Logs only if the score is reasonable (above 5) to avoid spamming
            if self.debug:
                marker = " <--- NEW BEST" if val > self.best_score else ""
                print(f"[AF] Pos: {self.current_pos} | Score: {val:.2f}{marker}")

            if val > self.best_score:
                self.best_score = val
                self.best_pos = self.current_pos

            if self.current_pos < self.scan_end:
                self.current_pos += self.step_size
                # Overflow protection
                if self.current_pos > self.MAX_FOCUS_VALUE:
                    self.current_pos = self.MAX_FOCUS_VALUE

                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                self.wait_counter = self.FRAMES_TO_WAIT
                return False, None
            else:
                # Scan finished.
                print(f">>> [AF] Peak found at {self.best_pos} with score {self.best_score:.2f}")

                # Sanity check: If score is too low, it's likely too dark or no object
                if self.best_score < 10.0:
                    print("!!! [WARNING] Image score implies low contrast or poor lighting !!!")

                # Switching to Backlash correction (precise return)
                self.stage = "backlash_dip"

                # Dipping 150 below target to return to it from below
                self.current_pos = max(0, self.best_pos - 150)
                print(f"[AF] Backlash Logic: Dipping to {self.current_pos}")

                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                self.wait_counter = 15  # Long descent time
                return False, None

        # -- Stage 3: Rising to Target (Backlash Fix) --
        if self.stage == "backlash_dip":
            print(f"[AF] Backlash Logic: RISING to target {self.best_pos}")
            self.focuser.set(Focuser.OPT_FOCUS, self.best_pos)
            self.stage = "done"
            self.wait_counter = 5
            return False, None

        if self.stage == "done":
            return True, self.best_pos

        return False, None