
import cv2
import time
import requests
import sys

BASE_URL = "http://10.0.0.26/stream"

def test_cv2(url, label):
    print(f"[{label}] Testing cv2.VideoCapture({url})...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if cap.isOpened():
        print(f"[{label}] SUCCESS: Opened.")
        ret, frame = cap.read()
        if ret:
            print(f"[{label}] SUCCESS: Read frame. Shape: {frame.shape}")
        else:
            print(f"[{label}] FAILURE: Opened but failed to read frame.")
    else:
        print(f"[{label}] FAILURE: Failed to open.")
    cap.release()

if __name__ == "__main__":
    print("--- SUFFIX DEBUG START ---")
    test_cv2(BASE_URL, "Original")
    test_cv2(BASE_URL + "?type=.mjpg", "Query .mjpg")
    test_cv2(BASE_URL + "#.mjpg", "Hash .mjpg")
    print("--- SUFFIX DEBUG END ---")
