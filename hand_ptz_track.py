import sys
import socket
import struct
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import queue
from PIL import Image, ImageTk
import mediapipe as mp

# ===== NETWORK & CAMERA CONFIGURATION =====
PI_IP = "192.168.1.168"
STREAM_PORT = 8000
CONTROL_PORT = 5005
# ==========================================

# Global variables
control_socket = None
frame_queue = queue.Queue()

# ===== Hand Detection Setup =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


# --- Send commands to Raspberry Pi ---
def send_command(command: str):
    global control_socket
    try:
        if control_socket:
            control_socket.sendall((command + '\n').encode('utf-8'))
            print(f"Sent command: {command}")
    except Exception as e:
        print(f"Failed to send command: {e}")
        messagebox.showerror("Connection Error", "Failed to send command. Check server connection.")


# --- Video stream from Raspberry Pi ---
def stream_video():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((PI_IP, STREAM_PORT))
        print("Video stream client connected.")
    except Exception as e:
        print(f"Could not connect to video stream: {e}")
        return

    connection = client_socket.makefile('rb')

    try:
        while True:
            image_len_bytes = connection.read(struct.calcsize('<L'))
            if not image_len_bytes:
                break

            image_len = struct.unpack('<L', image_len_bytes)[0]

            frame_data = b''
            while len(frame_data) < image_len:
                chunk = connection.read(image_len - len(frame_data))
                if not chunk:
                    break
                frame_data += chunk

            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is not None:
                frame_queue.put(frame)

    except Exception as e:
        print(f"Video stream stopped: {e}")

    finally:
        connection.close()
        client_socket.close()


# --- GUI Update Loop (with hand tracking) ---
def update_gui(video_label, root):
    if not frame_queue.empty():
        frame = frame_queue.get()

        # Hand detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2  # image center

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get wrist landmark
                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Compute offset from center
                dx, dy = x - cx, y - cy
                threshold = 50  # tolerance in pixels

                move_x, move_y = 0, 0
                if abs(dx) > threshold:
                    move_x = -5 if dx > 0 else 5  # left/right
                if abs(dy) > threshold:
                    move_y = 5 if dy > 0 else -5  # up/down

                if move_x or move_y:
                    send_command(f"move:{move_x}:{move_y}")

        # Show in Tkinter
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_label.configure(image=img_tk)
        video_label.image = img_tk

    root.after(10, update_gui, video_label, root)


# --- Control Panel (manual override) ---
def create_control_panel(root):
    style = ttk.Style(root)
    style.configure('TButton', font=('Helvetica', 12), padding=10)

    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    # --- Motor Controls ---
    motor_frame = ttk.LabelFrame(control_frame, text="Motor Control", padding=10)
    motor_frame.pack(pady=10, padx=10, fill="x")
    ttk.Button(motor_frame, text="Move Up", command=lambda: send_command("move:0:-5")).pack(fill="x", pady=2)
    ttk.Button(motor_frame, text="Move Down", command=lambda: send_command("move:0:5")).pack(fill="x", pady=2)
    ttk.Button(motor_frame, text="Move Left", command=lambda: send_command("move:5:0")).pack(fill="x", pady=2)
    ttk.Button(motor_frame, text="Move Right", command=lambda: send_command("move:-5:0")).pack(fill="x", pady=2)

    # --- Zoom and Focus Controls ---
    zoom_focus_frame = ttk.LabelFrame(control_frame, text="Zoom & Focus", padding=10)
    zoom_focus_frame.pack(pady=10, padx=10, fill="x")
    ttk.Button(zoom_focus_frame, text="Zoom In", command=lambda: send_command("zoom:100")).pack(fill="x", pady=2)
    ttk.Button(zoom_focus_frame, text="Zoom Out", command=lambda: send_command("zoom:-100")).pack(fill="x", pady=2)
    ttk.Button(zoom_focus_frame, text="Focus In", command=lambda: send_command("focus:5")).pack(fill="x", pady=2)
    ttk.Button(zoom_focus_frame, text="Focus Out", command=lambda: send_command("focus:-5")).pack(fill="x", pady=2)
    ttk.Button(zoom_focus_frame, text="Autofocus", command=lambda: send_command("autofocus")).pack(fill="x", pady=2)

    # --- Quit Button ---
    def on_quit():
        send_command("quit")
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_quit)
    ttk.Button(control_frame, text="Quit", command=on_quit).pack(pady=10)


# --- Main Program ---
def main():
    global control_socket

    # Connect to control server
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        control_socket.connect((PI_IP, CONTROL_PORT))
        print("Connected to control server.")
    except Exception as e:
        print(f"Could not connect to control server: {e}")
        messagebox.showerror("Connection Error", "Could not connect to control server.")
        return

    # Tkinter window
    root = tk.Tk()
    root.title("PTZ Camera Control with Hand Tracking")

    # Video area
    video_label = tk.Label(root)
    video_label.pack(side=tk.LEFT, padx=10, pady=10)

    # Video thread
    video_thread = threading.Thread(target=stream_video, daemon=True)
    video_thread.start()

    # Control panel
    create_control_panel(root)

    # GUI loop
    update_gui(video_label, root)
    root.mainloop()


if __name__ == "__main__":
    main()
