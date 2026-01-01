import cv2

def check_cameras():
    print("Searching for cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            ret, frame = cap.read()
            if ret:
                print(f"  - Read success: {frame.shape}")
            else:
                print("  - Read failed")
            cap.release()
        else:
            print(f"No camera at index {i}")

if __name__ == "__main__":
    check_cameras()
