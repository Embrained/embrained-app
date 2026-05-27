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
import matplotlib.pyplot as plt
import sys
import subprocess

# Ensure dependencies
try:
    import matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

def calculate_shift(img1, img2):
    """
    Calculate horizontal pixel shift minimizing SSD.
    Range: -50 to +50
    Returns integer shift (+ = Scene Left->Right = Camera Left)
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    h, w = gray1.shape
    h_start = int(h * 0.3) # Avoid top/bottom borders
    h_end = int(h * 0.7)
    
    # Use wider strip
    crop1 = gray1[h_start:h_end, :]
    crop2 = gray2[h_start:h_end, :]
    
    search_range = 60
    margin = search_range
    roi_w = w - 2*margin
    
    if roi_w <= 0: return 0
    
    ref_roi = crop1[:, margin : margin + roi_w]
    
    min_ssd = float('inf')
    best_shift = 0
    
    for dx in range(-search_range, search_range + 1):
        x_start = margin + dx
        tgt_roi = crop2[:, x_start : x_start + roi_w]
        
        diff = ref_roi.astype(float) - tgt_roi.astype(float)
        ssd = np.sum(diff**2)
        
        if ssd < min_ssd:
            min_ssd = ssd
            best_shift = dx
            
    return best_shift

def parse_cmd(cmd_str):
    try:
        # Format: l:-80;r:-80;
        parts = cmd_str.split(';')
        l_val = 0
        r_val = 0
        for p in parts:
            if 'l:' in p: l_val = int(p.replace('l:', ''))
            if 'r:' in p: r_val = int(p.replace('r:', ''))
        return l_val, r_val
    except:
        return 0, 0

def analyze_dataset(root_dir):
    start_frames = 0
    left_frames = 0
    right_frames = 0
    
    left_shifts = []
    right_shifts = []
    
    print(f"Scanning {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        if 'log.csv' not in files: continue
        
        log_path = os.path.join(root, 'log.csv')
        try:
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except:
            continue
            
        print(f"Processing {os.path.basename(root)} ({len(rows)} frames)...")
        
        prev_img = None
        
        for row in rows:
            img_name = row['img_file']
            img_path = os.path.join(root, "images", img_name)
            if not os.path.exists(img_path):
                # Try root
                img_path = os.path.join(root, img_name)
                
            curr_img = None
            if os.path.exists(img_path):
                curr_img = cv2.imread(img_path)
            
            if prev_img is not None and curr_img is not None:
                # Analyze Command
                l, r = parse_cmd(row['motor_cmd'])
                
                # Check for Stop
                if l == 0 and r == 0:
                    pass # Stop
                else: 
                    # Verify symmetric turn
                    if l == r:
                        # Standard Turn (Inverted logic or not, just grouping by sign)
                        # Case 1: L < 0, R < 0 (e.g. -80)
                        if l < 0:
                            if not left_shifts: # First one
                                cv2.imwrite("debug_pair_prev.png", prev_img)
                                cv2.imwrite("debug_pair_curr.png", curr_img)
                                print("DEBUG: Saved debug_pair_prev.png and debug_pair_curr.png")
                            
                            shift = calculate_shift(prev_img, curr_img)
                            # Only count significant shifts to reduce noise? 
                            # Or capture all to see distribution
                            left_shifts.append(shift)
                            left_frames += 1
                        # Case 2: L > 0, R > 0 (e.g. 80)
                        elif l > 0:
                            shift = calculate_shift(prev_img, curr_img)
                            right_shifts.append(shift)
                            right_frames += 1
                            
            prev_img = curr_img
            start_frames += 1
            
    # RESULTS
    print("\n" + "="*40)
    print("ANALYSIS RESULTS")
    print("="*40)
    
    if left_frames > 0:
        avg_l = sum(left_shifts) / len(left_shifts)
        pos_l = sum(1 for x in left_shifts if x > 0)
        neg_l = sum(1 for x in left_shifts if x < 0)
        print(f"Negative Command (L<0, R<0):")
        print(f"  Count: {left_frames}")
        print(f"  Avg Pixel Shift: {avg_l:.2f}")
        print(f"  Positive Shifts: {pos_l} ({pos_l/left_frames*100:.1f}%)")
        print(f"  Negative Shifts: {neg_l} ({neg_l/left_frames*100:.1f}%)")
    else:
        print("No Negative Command Frames Found.")

    print("-" * 40)

    if right_frames > 0:
        avg_r = sum(right_shifts) / len(right_shifts)
        pos_r = sum(1 for x in right_shifts if x > 0)
        neg_r = sum(1 for x in right_shifts if x < 0)
        print(f"Positive Command (L>0, R>0):")
        print(f"  Count: {right_frames}")
        print(f"  Avg Pixel Shift: {avg_r:.2f}")
        print(f"  Positive Shifts: {pos_r} ({pos_r/right_frames*100:.1f}%)")
        print(f"  Negative Shifts: {neg_r} ({neg_r/right_frames*100:.1f}%)")
    else:
        print("No Positive Command Frames Found.")
    
    print("="*40)

if __name__ == "__main__":
    target_dir = os.path.join(os.getcwd(), 'data', 'nook')
    analyze_dataset(target_dir)
