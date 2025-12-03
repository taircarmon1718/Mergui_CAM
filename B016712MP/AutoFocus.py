import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1200

    # הגדלנו דרמטית את ההמתנה: 10 פריימים (כ-0.3 שניות) לכל צעד
    # זה מבטיח שהתמונה שאנחנו מודדים היא באמת מהמיקום הנוכחי
    FRAMES_TO_WAIT = 10

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

    def get_sharpness(self, frame):
        # שימוש ב-Sobel רגיל שהוא פחות רגיש לרעש מ-Laplacian
        h, w = frame.shape[:2]

        # חיתוך מרכז מדויק (רבע תמונה)
        roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # טשטוש קל חובה
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

        abs_gx = cv2.convertScaleAbs(gx)
        abs_gy = cv2.convertScaleAbs(gy)

        score = cv2.addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0)
        return float(np.mean(score))

    def startFocus_hailo(self):
        # איפוס ל-0
        print("[AF] Resetting lens to 0...")
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.stage = "reset_wait"
        self.wait_counter = 15  # זמן ארוך לאיפוס מכני

    def stepFocus_hailo(self, frame):
        # -- מנגנון השהייה --
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return False, None

        # -- שלב 1: המתנה לאיפוס --
        if self.stage == "reset_wait":
            print("[AF] Starting COARSE scan...")
            self.stage = "scanning"
            self.current_pos = 0
            self.scan_end = self.MAX_FOCUS_VALUE
            self.step_size = 50  # קפיצות של 50
            self.best_score = -1
            return False, None

        # -- שלב 2: סריקה לינארית --
        if self.stage == "scanning":
            # מדידה
            val = self.get_sharpness(frame)

            if self.debug:
                is_best = " NEW BEST!" if val > self.best_score else ""
                print(f"[AF] Pos: {self.current_pos} | Score: {val:.2f}{is_best}")

            if val > self.best_score:
                self.best_score = val
                self.best_pos = self.current_pos

            # התקדמות
            if self.current_pos < self.scan_end:
                self.current_pos += self.step_size
                if self.current_pos > self.MAX_FOCUS_VALUE:
                    self.current_pos = self.MAX_FOCUS_VALUE

                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                self.wait_counter = self.FRAMES_TO_WAIT
                return False, None
            else:
                # סיימנו סריקה גסה. עכשיו יש לנו "אזור חשוד".
                # הבעיה: כנראה פספסנו את הנקודה המדויקת בגלל הקפיצות הגדולות
                print(f"[AF] Coarse done. Best approx pos: {self.best_pos}")

                # נגדיר טווח סריקה עדין סביב התוצאה
                start_fine = max(0, self.best_pos - 80)
                end_fine = min(self.MAX_FOCUS_VALUE, self.best_pos + 80)

                self.stage = "fine_scan"
                self.current_pos = start_fine
                self.scan_end = end_fine
                self.step_size = 10  # קפיצות קטנות מאוד
                self.best_score = -1  # מאפסים ניקוד כדי למצוא שיא מקומי מדויק

                # מזיזים להתחלה ומחכים
                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                self.wait_counter = 15
                return False, None

        # -- שלב 3: סריקה עדינה (FINE) --
        if self.stage == "fine_scan":
            val = self.get_sharpness(frame)

            # הדפסה רק אם הניקוד גבוה יחסית (כדי לא להציף לוגים)
            if self.debug and val > 10:
                print(f"[AF-FINE] Pos: {self.current_pos} | Score: {val:.2f}")

            if val > self.best_score:
                self.best_score = val
                self.best_pos = self.current_pos

            if self.current_pos < self.scan_end:
                self.current_pos += self.step_size
                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                # גם כאן - המתנה ארוכה יחסית לדיוק מקסימלי
                self.wait_counter = 6
                return False, None
            else:
                # סיימנו הכל!
                print(f">>> [AF] FINAL DECISION: Position {self.best_pos}")

                # תנועה סופית למטרה
                self.focuser.set(Focuser.OPT_FOCUS, self.best_pos)

                # אבל רגע! בגלל שזזנו קדימה, אולי עברנו אותה קצת (Backlash).
                # נעשה תיקון זעיר: נלך אחורה 50 צעדים ואז נחזור.
                self.stage = "correction_step_1"
                self.wait_counter = 5
                return False, None

        # -- שלב 4: תיקון Backlash אחרון --
        if self.stage == "correction_step_1":
            # יורדים קצת מתחת למטרה
            correction = max(0, self.best_pos - 100)
            print(f"[AF] Backlash correction: Dip to {correction}")
            self.focuser.set(Focuser.OPT_FOCUS, correction)
            self.stage = "correction_step_2"
            self.wait_counter = 10
            return False, None

        if self.stage == "correction_step_2":
            # עולים בול למטרה
            print(f"[AF] Backlash correction: Rise to {self.best_pos}")
            self.focuser.set(Focuser.OPT_FOCUS, self.best_pos)
            self.stage = "done"
            self.wait_counter = 5
            return False, None

        if self.stage == "done":
            return True, self.best_pos

        return False, None