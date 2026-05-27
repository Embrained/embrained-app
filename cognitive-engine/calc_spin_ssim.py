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

import os
import csv
import cv2
import numpy as np
import sys

# Try importing ssim
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    print("scikit-image not found. Computing MSE instead.")
    HAS_SSIM = False

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def calculate_closure(log_path, target_ts, duration_frames=25):
    dataset_dir = os.path.dirname(log_path)
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        images_dir = dataset_dir # Fallback to root
        
    print(f"Target Timestamp: {target_ts}")
    print(f"Dataset Dir: {dataset_dir}")
    
    rows = []
    try:
        with open(log_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Error reading log: {e}")
        return

    # Find row with closest timestamp
    start_index = -1
    min_diff = float('inf')
    
    # Quick scan by timestamp
    # Assuming rows are sorted, we could binary search, but linear is fine for 20k rows
    for i, row in enumerate(rows):
        try:
            ts = float(row['timestamp'])
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                start_index = i
        except:
            continue
            
    if start_index == -1 or min_diff > 0.5:
        print(f"Could not find exact timestamp. Closest diff: {min_diff}")
        return

    print(f"Found Start Index: {start_index} (Diff: {min_diff:.6f})")
    
    # Collect Images
    if start_index + duration_frames > len(rows):
        duration_frames = len(rows) - start_index
        
    image_data = [] # (path, ts)
    
    for i in range(duration_frames):
        row = rows[start_index + i]
        img_name = row['img_file']
        path = os.path.join(images_dir, img_name)
        if not os.path.exists(path):
            # Try flat structure
            path = os.path.join(dataset_dir, img_name)
            
        if os.path.exists(path):
            image_data.append((path, float(row['timestamp'])))
        else:
            print(f"Missing image: {img_name}")
            
    if not image_data:
        print("No images found.")
        return

    # Load Ref Image
    ref_path, ref_ts = image_data[0]
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print("Failed to load reference image.")
        return
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    print("\nCircular Closure Analysis:")
    print(f"Metric: {'SSIM' if HAS_SSIM else 'MSE (Lower is Better)'}")
    print(f"{'Offset':<6} | {'Time Delta':<10} | {'Score'}")
    print("-" * 40)
    
    # Compute
    for i in range(1, len(image_data)):
        curr_path, curr_ts = image_data[i]
        curr_img = cv2.imread(curr_path)
        if curr_img is None:
            continue
            
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        
        if HAS_SSIM:
            score, _ = ssim(ref_gray, curr_gray, full=True)
            # SSIM: Higher is better (1.0 = match)
        else:
            score = mse(ref_gray, curr_gray)
            # MSE: Lower is better (0.0 = match)
        
        dt = curr_ts - ref_ts
        print(f"+{i:<5} | {dt:.4f}s    | {score:.5f}")

if __name__ == "__main__":
    # VALID CANDIDATE: Capture 2025-05-24 02_06_38
    target_log = r"c:\Users\chris\Embrained\embrained-app\data\livingroom\capture-2025-05-24 02_06_38\log.csv"
    target_time = 1748111979.799
    
    calculate_closure(target_log, target_time, 30) # Look a bit further than 24 just in case
