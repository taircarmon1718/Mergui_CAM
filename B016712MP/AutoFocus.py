import cv2
import numpy as np
import time
from B016712MP.Focuser import Focuser

class AutoFocus:
    MAX_FOCUS_VALUE = 1200
    FRAMES_TO_WAIT = 4  # המתנה להתייצבות תמונה

    def __init__(self, focuser, camera=None, debug=False):
        self.focuser = focuser
        self.debug = debug
        
        self.stage = "idle"
        self.max_index = 0
        self.max_value = -1.0
        self.current_pos = 0
        self.wait_counter = 0
        
        # משתנים ללוגיקה
        self.scan_start = 0
        self.scan_end = 0
        self.step_size = 0

    def get_sharpness(self, frame):
        # שיטת השונות (Variance) עם טשטוש לסינון רעשים
        h, w = frame.shape[:2]
        roi = frame[h//3 : 2*h//3, w//3 : 2*w//3] # חיתוך שליש מרכזי
        
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0) # חובה לסינון רעש!
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def startFocus_hailo(self):
        # מתחילים תמיד מ-0 כדי "לאפס" את הגיר
        self.focuser.set(Focuser.OPT_FOCUS, 0)
        self.stage = "resetting"
        self.wait_counter = 10 # מחכים שהמנוע יגיע ל-0 בוודאות

    def stepFocus_hailo(self, frame):
        # 1. מנגנון השהייה (Debounce)
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return False, None

        # ---------------------------------------------------------
        # שלב 0: איפוס מכני
        # ---------------------------------------------------------
        if self.stage == "resetting":
            # סיימנו לחכות לאיפוס, מתחילים סריקה גסה
            print("[AF] Reset done. Starting Coarse Scan...")
            self.stage = "coarse"
            self.current_pos = 0
            self.scan_end = self.MAX_FOCUS_VALUE
            self.step_size = 40 # קפיצות של 40
            self.max_value = -1
            return False, None

        # ---------------------------------------------------------
        # שלב 1: סריקה (גם גסה וגם עדינה)
        # ---------------------------------------------------------
        if self.stage in ["coarse", "fine"]:
            # א. מדידה
            val = self.get_sharpness(frame)
            
            if self.debug:
                 print(f"[AF] {self.stage} | Pos: {self.current_pos} | Score: {val:.1f}")

            # ב. שמירת השיא
            if val > self.max_value:
                self.max_value = val
                self.max_index = self.current_pos

            # ג. התקדמות
            if self.current_pos < self.scan_end:
                self.current_pos += self.step_size
                if self.current_pos > self.MAX_FOCUS_VALUE:
                    self.current_pos = self.MAX_FOCUS_VALUE
                
                self.focuser.set(Focuser.OPT_FOCUS, self.current_pos)
                self.wait_counter = self.FRAMES_TO_WAIT
                return False, None
            
            else:
                # סיימנו את הטווח הנוכחי
                if self.stage == "coarse":
                    # עוברים לסריקה עדינה סביב השיא שנמצא
                    print(f">>> Peak found at {self.max_index}. Refining...")
                    
                    # הולכים אחורה כדי להתחיל את העדין מלמטה (חשוב!)
                    start_fine = max(0, self.max_index - 150)
                    self.scan_end = min(self.MAX_FOCUS_VALUE, self.max_index + 150)
                    
                    self.stage = "positioning_for_fine"
                    self.current_pos = start_fine
                    self.focuser.set(Focuser.OPT_FOCUS, start_fine)
                    self.wait_counter = 10 # נותנים למנוע זמן לרדת
                    return False, None
                
                elif self.stage == "fine":
                    # מצאנו את הפוקוס האמיתי!
                    print(f">>> FINAL BEST FOCUS at {self.max_index} (Score: {self.max_value})")
                    
                    # === התיקון הקריטי: Anti-Backlash ===
                    # במקום ללכת ישר לנקודה, אנחנו יורדים מתחתיה ועולים חזרה
                    target = self.max_index
                    overshoot_pos = max(0, target - 200) # יורדים הרבה למטה
                    
                    print(f"[AF] Applying Backlash Fix: Moving to {overshoot_pos} first...")
                    self.focuser.set(Focuser.OPT_FOCUS, overshoot_pos)
                    
                    self.stage = "anti_backlash_wait"
                    self.wait_counter = 15 # מחכים שהירידה תושלם
                    return False, None

        # ---------------------------------------------------------
        # שלב מעבר: הכנה לסריקה עדינה
        # ---------------------------------------------------------
        if self.stage == "positioning_for_fine":
            self.stage = "fine"
            self.step_size = 10 # צעדים קטנים
            self.max_value = -1 # מאפסים ניקוד כדי למצוא שיא מקומי מדויק
            return False, None

        # ---------------------------------------------------------
        # שלב סיום: חזרה לנקודה המדויקת (מלמטה למעלה)
        # ---------------------------------------------------------
        if self.stage == "anti_backlash_wait":
            # עכשיו אנחנו למטה (ב-overshoot_pos). זה הזמן לעלות בול למטרה.
            print(f"[AF] Moving UP to target: {self.max_index}")
            self.focuser.set(Focuser.OPT_FOCUS, self.max_index)
            self.stage = "done"
            # ניתן לו רגע להתייצב לפני שמחזירים True
            self.wait_counter = 5 
            return False, None
            
        if self.stage == "done":
            return True, self.max_index

        return False, None