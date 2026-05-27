# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
