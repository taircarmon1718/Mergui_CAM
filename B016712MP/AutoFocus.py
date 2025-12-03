import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1200

    # הגדלנו ל-4 כדי לוודא שאין רעידות בזמן הצילום
    FRAMES_TO_WAIT = 4

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.camera = camera
        self.debug = debug

        self.stage = "idle"
        self.step = 0
        self.stage_end = 0

        self.max_index = 0
        self.max_value = -1.0
        self.focal = 0

        self.wait_counter = 0

        # משתנים לטיפול בחזרה לנקודה
        self.target_focus = 0
        self.backlash_steps = 0

    # ============================================================
    #  Tenengrad - Center ROI
    # ============================================================
    def get_sharpness(self, frame):
        h, w = frame.shape[:2]
        # חיתוך ROI - שליש מרכזי
        cx, cy = w // 2, h // 2
        half_w, half_h = w // 6, h // 6
        roi = frame[cy - half_h: cy + half_h, cx - half_w: cx + half_w]

        # המרה לאפור
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # --- שלב קריטי: ניקוי רעשים ---
        # טשטוש קל מעלים "רעש" דיגיטלי שעלול להיראות כמו פוקוס
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # חישוב לפלסיאן
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # החזרת השונות (Variance) - ככל שהפיזור גדול יותר, התמונה חדה יותר
        score = laplacian.var()
        return float(score)

    # ============================================================
    # Init Stage
    # ============================================================
    def _init_stage(self, step, start_pos, end_pos, stage):
        if self.debug:
            print(f"[AF] Init {stage}: Start={start_pos}, End={end_pos}, Step={step}")

        self.stage = stage
        self.step = step

        start_pos = max(0, int(start_pos))
        end_pos = min(int(end_pos), self.MAX_FOCUS_VALUE)

        self.focal = start_pos
        self.stage_end = end_pos

        self.max_index = start_pos
        self.max_value = -1.0

        self.wait_counter = self.FRAMES_TO_WAIT
        self.focuser.set(Focuser.OPT_FOCUS, start_pos)

    def startFocus_hailo(self):
        # סריקה גסה
        self._init_stage(step=40, start_pos=0, end_pos=self.MAX_FOCUS_VALUE, stage="coarse")

    # ============================================================
    # Main Logic
    # ============================================================
    def stepFocus_hailo(self, frame):
        # אם סיימנו הכל
        if self.stage == "done":
            return True, self.max_index

        if self.stage == "idle":
            return False, None

        # --- מנגנון המתנה להתייצבות ---
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return False, None

        # --- שלב מיוחד: תיקון Backlash (חזרה לנקודה) ---
        if self.stage == "positioning":
            # הגענו לשלב החזרה. אנחנו כרגע בנקודת איפוס (למשל Best-150)
            # עכשיו עולים לנקודה האמיתית
            print(f">>> [AF] Positioning: Final move to {self.target_focus}")
            self.focuser.set(Focuser.OPT_FOCUS, self.target_focus)
            self.stage = "done"  # סיימנו! בפריים הבא נחזיר True

            # (אופציונלי) לתת עוד זמן התייצבות אם צריך, אבל זה סוף התהליך
            return False, None

        # --- מדידת חדות רגילה ---
        val = self.get_sharpness(frame)

        if self.debug:
            # מדפיס כוכבית אם זה שיא חדש
            marker = "***" if val > self.max_value else ""
            print(f"[AF] {self.stage} | Pos: {self.focal} | Val: {val:.2f} {marker}")

        if val > self.max_value:
            self.max_value = val
            self.max_index = self.focal

        # --- בדיקה אם סיימנו את הטווח הנוכחי ---
        if self.focal >= self.stage_end:

            if self.stage == "coarse":
                # סיימנו גס, עוברים לעדין
                fine_start = max(0, self.max_index - 120)
                fine_end = min(self.max_index + 120, self.MAX_FOCUS_VALUE)
                print(f">>> [AF] Coarse Peak: {self.max_index}. Refine: {fine_start}-{fine_end}")
                self._init_stage(step=10, start_pos=fine_start, end_pos=fine_end, stage="fine")
                return False, None

            elif self.stage == "fine":
                # סיימנו עדין. מצאנו את ה-Best האמיתי.
                print(f">>> [AF] Found BEST: {self.max_index} (Val: {self.max_value})")

                # טריק ה-Backlash:
                # כדי לנחות בול, אנחנו חייבים להגיע מלמטה.
                # נלך אחורה לנקודה נמוכה יותר, נחכה, ואז נעלה לנקודה הנכונה.

                reset_pos = max(0, self.max_index - 200)  # יורדים 200 צעדים מתחת למטרה
                self.target_focus = self.max_index

                print(f">>> [AF] Anti-Backlash: Resetting to {reset_pos} before going to {self.target_focus}")
                self.focuser.set(Focuser.OPT_FOCUS, reset_pos)

                # נעבור לסטטוס ביניים
                self.stage = "positioning"
                self.wait_counter = 8  # מחכים זמן ארוך (כ-0.25 שניות) שהמנוע ירד למטה

                return False, None

        # --- התקדמות בסריקה ---
        self.focal += self.step
        if self.focal > self.stage_end:
            self.focal = self.stage_end

        self.focuser.set(Focuser.OPT_FOCUS, self.focal)
        self.wait_counter = self.FRAMES_TO_WAIT

        return False, None