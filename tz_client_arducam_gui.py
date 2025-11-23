import threading, time, requests, tkinter as tk
from tkinter import ttk, messagebox
import cv2

PI_IP = "192.168.1.168"                 # <-- עדכני ל-IP של ה-Pi
PTZ_PORT = 5005
RTSP_URL = "rtsp://localhost:8554/horse"
STEP_MOTOR, STEP_FOCUS, STEP_ZOOM = 5, 5, 100

BASE = f"http://{PI_IP}:{PTZ_PORT}"
def post(path, params=None):
    try:
        r = requests.post(f"{BASE}{path}", params=params, timeout=2); r.raise_for_status(); return r.json()
    except Exception as e: messagebox.showerror("HTTP Error", str(e)); return None
def get_(path):
    try:
        r = requests.get(f"{BASE}{path}", timeout=2); r.raise_for_status(); return r.json()
    except Exception as e: messagebox.showerror("HTTP Error", str(e)); return None

class PTZGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Arducam PTZ Controller + Live View"); self.geometry("420x300")
        row=0
        ttk.Button(self,text="Center",command=lambda:post("/center")).grid(row=row,column=0,padx=5,pady=5)
        ttk.Button(self,text="Toggle Mode",command=lambda:post("/mode")).grid(row=row,column=1,padx=5,pady=5)
        ttk.Button(self,text="IR-CUT",command=lambda:post("/ircut")).grid(row=row,column=2,padx=5,pady=5)
        ttk.Button(self,text="Autofocus",command=lambda:post("/autofocus")).grid(row=row,column=3,padx=5,pady=5); row+=1
        ttk.Label(self,text="Focus").grid(row=row,column=0,sticky="e")
        ttk.Button(self,text="-",width=3,command=lambda:post("/focus",{"step":-STEP_FOCUS})).grid(row=row,column=1)
        ttk.Button(self,text="+",width=3,command=lambda:post("/focus",{"step":+STEP_FOCUS})).grid(row=row,column=2); row+=1
        ttk.Label(self,text="Zoom").grid(row=row,column=0,sticky="e")
        ttk.Button(self,text="-",width=3,command=lambda:post("/zoom",{"step":-STEP_ZOOM})).grid(row=row,column=1)
        ttk.Button(self,text="+",width=3,command=lambda:post("/zoom",{"step":+STEP_ZOOM})).grid(row=row,column=2); row+=1
        f=ttk.Frame(self); f.grid(row=row,column=0,columnspan=4,pady=10)
        ttk.Button(f,text="◀",width=5,command=lambda:post("/step",{"dx":+STEP_MOTOR})).grid(row=1,column=0,padx=5)
        ttk.Button(f,text="▲",width=5,command=lambda:post("/step",{"dy":-STEP_MOTOR})).grid(row=0,column=1,padx=5)
        ttk.Button(f,text="▼",width=5,command=lambda:post("/step",{"dy":+STEP_MOTOR})).grid(row=2,column=1,padx=5)
        ttk.Button(f,text="▶",width=5,command=lambda:post("/step",{"dx":-STEP_MOTOR})).grid(row=1,column=2,padx=5)
        self.bind("<Left>",  lambda e: post("/step", {"dx": +STEP_MOTOR}))
        self.bind("<Right>", lambda e: post("/step", {"dx": -STEP_MOTOR}))
        self.bind("<Up>",    lambda e: post("/step", {"dy": -STEP_MOTOR}))
        self.bind("<Down>",  lambda e: post("/step", {"dy": +STEP_MOTOR}))
        self.bind("<Escape>", lambda e: self.destroy())
        threading.Thread(target=self.video_loop, daemon=True).start()
        print(get_("/status"))
    def video_loop(self):
        cap=cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened(): print("Failed to open RTSP"); return
        while True:
            ok,frame=cap.read()
            if not ok: time.sleep(0.05); continue
            cv2.imshow("Live View (RTSP)", frame)
            if cv2.waitKey(1)&0xFF==27: break
        cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    PTZGui().mainloop()
