import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1200

    # המתנה נדיבה להתייצבות
    FRAMES_TO_WAIT = 8

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.debug = debug

        self.stage = "idle"
        self.best_pos = 0
        self.best_score = -1.0
        self.current_pos = 0
        self.wait_counter = 0

        # טווח סריקה
        self.scan_start = 0
        self.scan_end = 0
        self.step_size = 0

    # =================================================================
    # הלב של האלגוריתם - חישוב חדות משופר
    # =================================================================
    def get_sharpness(self, frame):
        h, w = frame.shape[:2]

        # 1. חיתוך אגרסיבי למרכז (רק האובייקט חשוב)
        # לוקחים ריבוע קטן באמצע (1/6 מהרוחב)
        roi = frame[h // 2 - h // 8: h // 2 + h // 8, w // 2 - w // 8: w // 2 + w // 8]

        # 2. המרה לאפור
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # 3. הגברת ניגודיות (Histogram Equalization) - קריטי!
        # זה הופך אפור-משעמם לשחור-לבן חזק, ומבליט קווים
        gray = cv2.equalizeHist(gray)

        # 4. סינון רעשים (Gaussian Blur)
        # זה מוחק את ה"שלג" כדי שלא נתפקס עליו
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 5. זיהוי קצוות עם Sobel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 6. חישוב עוצמת הקצוות (Magnitude)
        mag = cv2.magnitude(gx, gy)

        # 7. סף רעש (Threshold) - הטריק החשוב ביותר
        # אנחנו מתעלמים מכל "קו" שהוא חלש מדי (מתחת ל-50).
        # זה מבטיח שנמדוד רק קצוות אמיתיים ולא רעש רקע.
        _, mag = cv2.threshold(mag, 50, 255, cv2.THRESH_TOZERO)

        # מחזירים את ממוצע הקווים החזקים
        return float(np.mean(mag))

    # =================================================================
    # ניהול התהליך
    # =================================================================
    def startFocus_hailo(self):
        print("[AF] Resetting lens to 0...")
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.stage = "reset_wait"
        self.wait_counter = 15

    def stepFocus_hailo(self, frame):
        # מנגנון השהייה
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return False, None

        # -- שלב 1: התחלה --
        if self.stage == "reset_wait":
            print("[AF] Starting Scan...")
            self.stage = "scanning"
            self.current_pos = 0
            self.scan_end = self.MAX_FOCUS_VALUE
            # צעדים קטנים יותר מההתחלה כדי לא לפספס פיקים אמיתיים
            self.step_size = 25
            self.best_score = -1
            self.best_pos = 0
            return False, None

        # -- שלב 2: סריקה לינארית --
        if self.stage == "scanning":
            val = self.get_sharpness(frame)

            # לוגים רק אם הציון סביר (מעל 5) כדי לא להספים
            if self.debug:
                marker = " <--- NEW BEST" if val > self.best_score else ""
                print(f"[AF] Pos: {self.current_pos} | Score: {val:.2f}{marker}")

            if val > self.best_score:
                self.best_score = val
                self.best_pos = self.current_pos

            if self.current_pos < self.scan_end:
                self.current_pos += self.step_size
                # הגנה מחריגה
                if self.current_pos > self.MAX_FOCUS_VALUE:
                    self.current_pos = self.MAX_FOCUS_VALUE

                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                self.wait_counter = self.FRAMES_TO_WAIT
                return False, None
            else:
                # סיימנו סריקה.
                print(f">>> [AF] Peak found at {self.best_pos} with score {self.best_score:.2f}")

                # בדיקת שפיות: אם הציון נמוך מדי, כנראה חשוך מדי או שאין אובייקט
                if self.best_score < 10.0:
                    print("!!! [WARNING] Image score implies low contrast or poor lighting !!!")

                # עוברים לתיקון Backlash (חזרה מדויקת)
                self.stage = "backlash_dip"

                # יורדים 150 מתחת למטרה כדי לחזור אליה מלמטה
                self.current_pos = max(0, self.best_pos - 150)
                print(f"[AF] Backlash Logic: Dipping to {self.current_pos}")

                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                self.wait_counter = 15  # זמן ירידה ארוך
                return False, None

        # -- שלב 3: עלייה למטרה (Backlash Fix) --
        if self.stage == "backlash_dip":
            print(f"[AF] Backlash Logic: RISING to target {self.best_pos}")
            self.focuser.set(Focuser.OPT_FOCUS, self.best_pos)
            self.stage = "done"
            self.wait_counter = 5
            return False, None

        if self.stage == "done":
            return True, self.best_pos

        return False, None