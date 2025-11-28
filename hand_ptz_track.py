import cv2
import os
import sys
import time

# הגדרת URL ה-RTSP שלך
RTSP_URL = "rtsp://192.168.1.168:8554/stream"
WINDOW_NAME = "Raspberry Pi Low Latency Stream"


def low_latency_rtsp_client(url):
    """
    מתחבר לזרם RTSP באמצעות מספר ניסיונות עם פרוטוקולים שונים
    ומציג אותו בחלון OpenCV.
    """

    # ניסיון ראשון: TCP (יציב יותר לחיבור הראשוני ופקודת DESCRIBE)
    # buffer_size: 1024 הוא המינימום היעיל ביותר
    # rtsp_flags: allow_only_interleaved_transport מכריח TCP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"Error: Initial TCP connection failed for {url}")
        print("Trying again with UDP transport...")
        cap.release()

        # ניסיון שני: UDP (מהיר יותר אך פחות יציב)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("Error: Could not open video stream using TCP or UDP.")
            print("Please check the IP address, port, and ensure the VLC server on the Pi is running without errors.")
            sys.exit(1)

    # הגדרת מאפיינים פנימיים במידת האפשר
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 0) # ניסיון לבטל את המאגר הפנימי

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print(f"Stream connected successfully at: {url}")
    print("Press 'q' to exit.")

    while True:
        # קורא פריים
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame. Retrying in 1 second...")
            time.sleep(1)  # ממתין ומנסה שוב

            # אם יש שגיאה, ננסה לאתחל מחדש את החיבור
            cap.release()
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("Reconnection failed. Exiting.")
                break
            continue

        # הצגת הפריים
        cv2.imshow(WINDOW_NAME, frame)

        # אם לוחצים על 'q', יוצאים מהלולאה
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # שחרור המשאבים
    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")


if __name__ == "__main__":
    low_latency_rtsp_client(RTSP_URL)