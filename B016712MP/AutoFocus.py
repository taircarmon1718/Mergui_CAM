import cv2
import numpy as np
from B016712MP.Focuser import Focuser


class AutoFocus:
    MAX_FOCUS_VALUE = 1200

    # כמה פריימים לחכות בין הזזה למדידה?
    # (3 זה בדרך כלל המספר בטוח ל-30fps כדי לוודא שהתמונה עדכנית)
    FRAMES_TO_WAIT = 3

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

        # מונה להמתנה בין תזוזות
        self.wait_counter = 0

    # ============================================================
    #  Tenengrad - Optimized for Center ROI
    # ============================================================
    def get_sharpness(self, frame):
        """
        חותך את המרכז (33% מהתמונה) ומחשב חדות רק עליו.
        זה הרבה יותר מהיר ומדויק לאובייקטים.
        """
        h, w = frame.shape[:2]

        # חיתוך המרכז (ROI) - שליש מהתמונה
        cx, cy = w // 2, h // 2
        half_w, half_h = w // 6, h // 6

        roi = frame[cy - half_h: cy + half_h, cx - half_w: cx + half_w]

        # המרת צבע וחישוב מהיר
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Tenengrad מהיר
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)  # 16S מהיר מ-32F
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

        # חישוב Magnitude בערך מוחלט (מהיר יותר מ-pow)
        abs_gx = cv2.convertScaleAbs(gx)
        abs_gy = cv2.convertScaleAbs(gy)

        score = cv2.addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0)
        return float(np.mean(score))

    # ============================================================
    # Initialize Stage
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

        # איפוס מונה המתנה
        self.wait_counter = self.FRAMES_TO_WAIT

        # תזוזה ראשונה
        self.focuser.set(Focuser.OPT_FOCUS, start_pos)

    # ============================================================
    # Start Logic
    # ============================================================
    def startFocus_hailo(self):
        # סריקה גסה: קפיצות של 50 (יותר מדויק מ-80)
        self._init_stage(step=50, start_pos=0, end_pos=self.MAX_FOCUS_VALUE, stage="coarse")

    # ============================================================
    # Main Loop (Call per frame)
    # ============================================================
    def stepFocus_hailo(self, frame):
        if self.stage in ("idle", "done"):
            return self.stage == "done", (self.max_index if self.stage == "done" else None)

        # --- מנגנון המתנה (De-bouncing) ---
        # אם הזזנו את המנוע, אנחנו חייבים לחכות כמה פריימים
        # כדי שהתמונה שמגיעה תתאים למיקום החדש של העדשה
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return False, None

        # --- מדידה ---
        val = self.get_sharpness(frame)

        if self.debug:
            print(f"[AF] {self.stage} | Pos: {self.focal} | Val: {val:.2f}")

        # שמירת המקסימום
        if val > self.max_value:
            self.max_value = val
            self.max_index = self.focal

        # --- החלטה האם להמשיך ---
        if self.focal >= self.stage_end:
            # סיימנו את השלב הנוכחי

            if self.stage == "coarse":
                # מעבר לסריקה עדינה סביב השיא שנמצא
                # טווח של +/- 100 סביב המקסימום הגס
                fine_start = max(0, self.max_index - 100)
                fine_end = min(self.max_index + 100, self.MAX_FOCUS_VALUE)

                print(f">>> [AF] Coarse Peak at {self.max_index}. Starting FINE scan...")

                self._init_stage(step=10, start_pos=fine_start, end_pos=fine_end, stage="fine")
                return False, None

            else:  # stage == "fine" -> סיימנו לגמרי
                print(f">>> [AF] DONE! Best Position: {self.max_index} (Score: {self.max_value:.2f})")
                self.focuser.set(Focuser.OPT_FOCUS, self.max_index)
                self.stage = "done"
                return True, self.max_index

        # --- התקדמות לצעד הבא ---
        self.focal += self.step

        # הגנה מחריגה
        if self.focal > self.stage_end:
            self.focal = self.stage_end  # בדיקה אחרונה בדיוק בקצה

        # ביצוע ההזזה הפיזית
        self.focuser.set(Focuser.OPT_FOCUS, self.focal)

        # איפוס המונה - חכה X פריימים עד המדידה הבאה
        self.wait_counter = self.FRAMES_TO_WAIT

        return False, None