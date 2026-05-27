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
import re

# Ensure dependencies
try:
    import natsort
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "natsort"])
    import natsort

def parse_cmd(cmd_str):
    try:
        parts = cmd_str.split(';')
        l_val = 0
        r_val = 0
        for p in parts:
            if 'l:' in p: l_val = int(p.replace('l:', ''))
            if 'r:' in p: r_val = int(p.replace('r:', ''))
        return l_val, r_val
    except:
        return 0, 0

def get_optical_flow_shift(img1, img2):
    """
    Calculate dense optical flow (Farneback) and return the average horizontal shift.
    """
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                        pyr_scale=0.5, levels=3, winsize=30, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # flow[..., 0] is horizontal (dx), flow[..., 1] is vertical (dy)
    avg_dx = np.mean(flow[..., 0])
    return avg_dx

def analyze_triggers(root_dir):
    print(f"Scanning {root_dir}...")
    
    left_triggers = []  # List of shifts for Left Triggers
    right_triggers = [] # List of shifts for Right Triggers
    
    CMD_THRESHOLD = 40
    LOOKAHEAD = 3 # Look at t+3
    
    capture_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('capture')]
    
    for capture_dir in capture_dirs:
        full_path = os.path.join(root_dir, capture_dir)
        log_path = os.path.join(full_path, 'log.csv')
        
        if not os.path.exists(log_path): continue
        
        print(f"Processing {capture_dir}...")
        
        # 1. Load Data
        rows = []
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows: continue
        
        # 2. Natural Sort Images (Critical for temporal continuity)
        # We need to map filename -> index in rows list to random access
        # But actually rows are usually chronological. 
        # Let's rely on timestamps in filenames? Or just trust rows order if we validate filenames?
        # The user emphasized checking numeric sorting.
        # Let's re-sort rows based on image filename timestamp.
        
        def extract_timestamp(row):
            fname = row['img_file']
            # Expected: 2026-01-10 13_12_25-1768068745397.jpg
            # Extract the last number part
            match = re.search(r'-(\d+)\.jpg', fname)
            if match:
                return int(match.group(1))
            return 0
            
        rows.sort(key=extract_timestamp)
        
        # 3. Iterate and Find Triggers
        for i in range(1, len(rows) - LOOKAHEAD):
            prev_row = rows[i-1]
            curr_row = rows[i]
            
            l_prev, r_prev = parse_cmd(prev_row['motor_cmd'])
            l_curr, r_curr = parse_cmd(curr_row['motor_cmd'])
            
            # Detect Start of Action (Stop -> Start)
            is_prev_stop = abs(l_prev) < CMD_THRESHOLD and abs(r_prev) < CMD_THRESHOLD
            is_curr_turn = abs(l_curr) > CMD_THRESHOLD # Just check magnitude first
            
            if is_prev_stop and is_curr_turn:
                # Validate it's a symmetric turn
                if l_curr == r_curr: # Standard turn
                    # Determine Direction
                    direction = "LEFT" if l_curr < 0 else "RIGHT" # Based on e.g. -80 vs 80
                    
                    # 4. Lookahead Measurement
                    frame_t_path = os.path.join(full_path, "images", curr_row['img_file'])
                    if not os.path.exists(frame_t_path): frame_t_path = os.path.join(full_path, curr_row['img_file'])
                    
                    frame_target_row = rows[i + LOOKAHEAD] # t + 3
                    frame_target_path = os.path.join(full_path, "images", frame_target_row['img_file'])
                    if not os.path.exists(frame_target_path): frame_target_path = os.path.join(full_path, frame_target_row['img_file'])
                    
                    if os.path.exists(frame_t_path) and os.path.exists(frame_target_path):
                        img_t = cv2.imread(frame_t_path)
                        img_target = cv2.imread(frame_target_path)
                        
                        if img_t is not None and img_target is not None:
                            shift = get_optical_flow_shift(img_t, img_target)
                            
                            if direction == "LEFT":
                                left_triggers.append(shift)
                            else:
                                right_triggers.append(shift)

    # 5. Reporting
    print("\n" + "="*40)
    print("LOOKAHEAD ANALYSIS RESULTS (Delta=+3 frames)")
    print("="*40)
    
    def analyze_group(shifts, name):
        if not shifts:
            print(f"{name}: No triggers found.")
            return
        avg = np.mean(shifts)
        median = np.median(shifts)
        std = np.std(shifts)
        count = len(shifts)
        # Expected Polarity
        # Left Command (L:-80) -> Scene moves Right (+ shift)
        # Right Command (L:80) -> Scene moves Left (- shift)
        pos_pct = np.sum(np.array(shifts) > 0.5) / count * 100
        neg_pct = np.sum(np.array(shifts) < -0.5) / count * 100
        
        print(f"{name} (Count: {count}):")
        print(f"  Avg Shift: {avg:.2f} px")
        print(f"  Median   : {median:.2f} px")
        print(f"  Std Dev  : {std:.2f}")
        print(f"  > +0.5px : {pos_pct:.1f}%")
        print(f"  < -0.5px : {neg_pct:.1f}%")

    analyze_group(left_triggers, "Negative Command (L<-40)")
    analyze_group(right_triggers, "Positive Command (L>40)")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(left_triggers, bins=30, alpha=0.5, label='Negative Cmd (L<0)', color='blue')
    plt.hist(right_triggers, bins=30, alpha=0.5, label='Positive Cmd (L>0)', color='red')
    plt.xlabel('Horizontal Pixel Shift (Optical Flow avg)')
    plt.ylabel('Frequency')
    plt.title(f'Pixel Shift Distribution (t vs t+{LOOKAHEAD}) for Turn Triggers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lookahead_report.png')
    print("\nSaved histogram to lookahead_report.png")

if __name__ == "__main__":
    target_dir = os.path.join(os.getcwd(), 'data', 'nook')
    try:
        analyze_triggers(target_dir)
    except KeyboardInterrupt:
        print("Interrupted.")
