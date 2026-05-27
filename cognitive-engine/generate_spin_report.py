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
import subprocess
import matplotlib.pyplot as plt
import math

# Ensure dependencies
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.metrics import structural_similarity as ssim

try:
    import matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

def shift_tolerant_ssim(ref_gray, tgt_gray, win=10):
    """
    Compute max SSIM by shifting tgt_gray within [-win, +win] in x and y.
    Robust to small vibrations/misalignments.
    """
    h, w = ref_gray.shape
    best_score = -1.0
    
    # We crop the reference by 'win' to avoid boundary issues during shift comparison
    # ref_roi = ref_gray[win:-win, win:-win]
    # Actually simpler: compare central crop of Ref to shifting crops of Tgt
    
    # If image is small, reduce win
    if win * 2 >= h or win * 2 >= w:
        win = 1
        
    roi_h = h - 2*win
    roi_w = w - 2*win
    
    # Center ROI of Reference
    ref_roi = ref_gray[win:win+roi_h, win:win+roi_w]
    
    # Grid search for best shift
    # Step size 2 for speed? Let's do 1 for precision on small sets
    for dy in range(-win, win+1, 2):
        for dx in range(-win, win+1, 2):
            # Extract shifted crop from Target
            y_start = win + dy
            x_start = win + dx
            tgt_roi = tgt_gray[y_start:y_start+roi_h, x_start:x_start+roi_w]
            
            score, _ = ssim(ref_roi, tgt_roi, full=True)
            if score > best_score:
                best_score = score
                
    return best_score

def find_intermittent_spins(root_dir):
    spin_events = []
    CMD_LEFT = "l:-100;r:-100;"
    CMD_RIGHT = "l:100;r:100;"
    CMD_STOP = "l:0;r:0;"
    MIN_ACTIVE = 5

    print(f"Scanning {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'log.csv':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if not rows: continue

                        current_type = None
                        start_idx = -1
                        active_count = 0
                        
                        i = 0
                        while i < len(rows):
                            row = rows[i]
                            cmd = row['motor_cmd'].strip()
                            
                            # Start Condition
                            if current_type is None:
                                is_left = (cmd == CMD_LEFT)
                                is_right = (cmd == CMD_RIGHT)
                                
                                if is_left or is_right:
                                    current_type = 'Left' if is_left else 'Right'
                                    start_idx = i
                                    active_count = 1
                                i += 1
                            else:
                                # Continue Condition
                                target_cmd = CMD_LEFT if current_type == 'Left' else CMD_RIGHT
                                
                                if cmd == target_cmd:
                                    active_count += 1
                                    i += 1
                                elif cmd == CMD_STOP:
                                    # Allowed pause, just continue
                                    i += 1
                                else:
                                    # Break Condition (Forward, Reverse, or Opposite Turn)
                                    end_idx = i - 1
                                    
                                    if active_count >= MIN_ACTIVE:
                                        # Refine Boundaries: Include 1 STOP before and 1 STOP after if available
                                        final_start = max(0, start_idx - 1)
                                        final_end = min(len(rows)-1, end_idx + 1)
                                        
                                        # Verify "pre" was a stop if processed? No, just include context frame.
                                        
                                        spin_events.append({
                                            'file': file_path,
                                            'type': current_type,
                                            'start_idx': final_start,
                                            'end_idx': final_end,
                                            'active': active_count,
                                            'start_ts': float(rows[start_idx]['timestamp'])
                                        })
                                    
                                    current_type = None
                                    # Do NOT skip this index, re-eval as potential new block start
                                    # But wait, we increment i manually. 
                                    continue
                        
                        # EOF Check
                        if current_type is not None:
                            end_idx = len(rows) - 1
                            if active_count >= MIN_ACTIVE:
                                final_start = max(0, start_idx - 1)
                                final_end = end_idx
                                spin_events.append({
                                    'file': file_path,
                                    'type': current_type,
                                    'start_idx': final_start,
                                    'end_idx': final_end,
                                    'active': active_count,
                                    'start_ts': float(rows[start_idx]['timestamp'])
                                })

                except Exception: pass

    spin_events.sort(key=lambda x: x['active'], reverse=True)
    return spin_events[:10]

def analyze_session(session, session_id):
    print(f"Analyzing Session {session_id}: {session['file']}...")
    
    # Load rows
    rows = []
    with open(session['file'], 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        rows = all_rows[session['start_idx'] : session['end_idx'] + 1]

    dataset_dir = os.path.dirname(session['file'])
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir): images_dir = dataset_dir

    # Load images & actions
    frames = [] # list of {img, cmd, ts, idx}
    
    for i, row in enumerate(rows):
        img_name = row['img_file']
        p = os.path.join(images_dir, img_name)
        if not os.path.exists(p): p = os.path.join(dataset_dir, img_name)
        
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                # Resize for speed/display? Keep original for SSIM calc
                frames.append({
                    'img': img,
                    'cmd': row['motor_cmd'],
                    'ts': float(row['timestamp']),
                    'idx': i
                })

    if not frames:
        print("No images found for session.")
        return

    # SSIM Calculation (Shift Tolerant)
    ref_gray = cv2.cvtColor(frames[0]['img'], cv2.COLOR_BGR2GRAY)
    scores = []
    
    print(f"  Computing SSIM for {len(frames)} frames...")
    for f_item in frames:
        gray = cv2.cvtColor(f_item['img'], cv2.COLOR_BGR2GRAY)
        
        # Frame 0 is ref
        if f_item['idx'] == 0:
            score = 1.0
        else:
            score = shift_tolerant_ssim(ref_gray, gray, win=10) # 10px search window
            
        scores.append(score)
        f_item['ssim'] = score

    # Find Key Indices
    scores_np = np.array(scores)
    trough_idx = np.argmin(scores_np)
    
    # Recovery: Max score AFTER trough
    if trough_idx < len(scores) - 1:
        recovery_slice = scores_np[trough_idx+1:]
        recovery_offset = np.argmax(recovery_slice)
        recovery_idx = trough_idx + 1 + recovery_offset
    else:
        recovery_idx = len(scores) - 1

    # ==========================
    # FILMSTRIP VISUALIZATION
    # ==========================
    num_frames = len(frames)
    cols = 5
    rows_grid = math.ceil(num_frames / cols)
    
    # Figure Size: 3 inches per col, 3 inches per row + 2 inches for plot
    fig_h = (rows_grid * 2.5) + 3 
    fig = plt.figure(figsize=(15, fig_h))
    
    plt.suptitle(f"Session {session_id} ({session['type']}) - Dur: {num_frames*0.1:.1f}s - Closure: {scores[recovery_idx]:.3f}", fontsize=16, y=0.99)

    # 1. Top Plot (Full Width)
    ax_plot = plt.subplot2grid((rows_grid + 1, cols), (0, 0), colspan=cols)
    ax_plot.plot(scores, label='Shift-Tolerant SSIM', color='black', linewidth=1.5)
    ax_plot.scatter(range(len(scores)), scores, s=15, color='gray') # Dots for frames
    
    # Highlight Key Points
    ax_plot.scatter([0], [scores[0]], color='blue', s=150, zorder=5, edgecolors='white')
    ax_plot.scatter([trough_idx], [scores[trough_idx]], color='red', s=150, zorder=5, edgecolors='white')
    ax_plot.scatter([recovery_idx], [scores[recovery_idx]], color='lime', s=150, zorder=5, edgecolors='white')
    
    ax_plot.set_ylabel('SSIM')
    ax_plot.set_xlabel('Frame Offset')
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_ylim(0, 1.05)

    # 2. Grid Images
    for i, frame in enumerate(frames):
        row_pos = (i // cols) + 1 # +1 to skip plot row
        col_pos = i % cols
        ax = plt.subplot2grid((rows_grid + 1, cols), (row_pos, col_pos))
        
        # Color Code Border
        border_col = 'gray'
        title_w = 'normal'
        if i == 0: border_col = 'blue'; title_w = 'bold'
        elif i == trough_idx: border_col = 'red'; title_w = 'bold'
        elif i == recovery_idx: border_col = 'lime'; title_w = 'bold'
        
        # Display Image
        ax.imshow(cv2.cvtColor(frame['img'], cv2.COLOR_BGR2RGB))
        
        # Add Border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3 if border_col != 'gray' else 1)
            
        # Simplified Command Label
        cmd = frame['cmd'].replace("l:","").replace("r:","").replace(";","")
        if "00" not in cmd and "0" in cmd: cmd = "STOP"
        elif "100" in cmd and "-100" not in cmd: cmd = "FWD" if "l:100" in frame['cmd'] and "r:100" in frame['cmd'] else cmd # Wait standard logic?
        # Just truncate
        if len(cmd) > 15: cmd = "..."
        
        ax.set_title(f"#{i} | {cmd}\nSSIM: {frame['ssim']:.3f}", fontsize=9, color='black', fontweight=title_w)
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = os.path.join("analysis_results", f"spin_report_v2_{session_id}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    target_dir = os.path.join(os.getcwd(), 'data', 'livingroom')
    top_sessions = find_intermittent_spins(target_dir)
    
    print(f"Found {len(top_sessions)} sessions. Generating refined reports...")
    
    for i, sess in enumerate(top_sessions):
        try:
            analyze_session(sess, i+1)
        except Exception as e:
            print(f"Failed session {i+1}: {e}")
