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
import sys
import json
import random
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OpticalFlowCheck")

# Configuration
DATASET_NAME = "nook"
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", DATASET_NAME)
TRANSITIONS_PATH = os.path.join(DATA_ROOT, 'all_transitions.json')
OUTPUT_HTML = "optical_flow_report.html"
OUTPUT_PLOT = "optical_flow_violin.png"
OUTPUT_SCATTER = "optical_flow_scatter.png"

IMG_H = 64
IMG_W = 64

def discretize_action(left, right):
    """
    Maps motor commands to Action IDs.
    Phys FWD  (0) <- l:-S, r:S
    Phys LEFT (1) <- l:-S, r:-S (Turning Left -> Scene moves Right)
    Phys RIGHT(2) <- l:S, r:S   (Turning Right -> Scene moves Left)
    Phys STOP (3) <- l:0, r:0
    Phys BACK (4) <- l:S, r:-S
    """
    tol = 1
    if abs(left) < 1 and abs(right) < 1: return 3 # STOP
    if left < -tol and right > tol: return 0 # FWD
    if left < -tol and right < -tol: return 1 # LEFT
    if left > tol and right > tol: return 2 # RIGHT
    if left > tol and right < -tol: return 4 # BACK
    return 3 # Default

def load_image_gray(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_W, IMG_H))
    return img

def main():
    logger.info("Starting Visual-Motor Alignment Check (Optical Flow)...")
    
    if not os.path.exists(TRANSITIONS_PATH):
        logger.error(f"Transitions file not found: {TRANSITIONS_PATH}")
        return

    # 1. Load Data
    logger.info(f"Loading transitions from {TRANSITIONS_PATH}...")
    with open(TRANSITIONS_PATH, 'r') as f:
        all_data = json.load(f)

    # Group by session
    sessions = {}
    for item in all_data:
        sess = item['session']
        if sess not in sessions: sessions[sess] = []
        sessions[sess].append(item)

    trajectories = []
    for sess, items in sessions.items():
        trajectories.append(sorted(items, key=lambda x: x['timestamp']))
    
    logger.info(f"Loaded {len(trajectories)} trajectories.")

    # 2. Sampling & Flow Calculation
    TARGET_PER_CLASS = 50
    counts = {1: 0, 2: 0, 3: 0, 0:0, 4:0}
    results = [] 
    
    logger.info(f"Sampling balanced pairs (Target {TARGET_PER_CLASS} per action)...")
    
    attempts = 0
    saved_debug = False
    
    while (counts[1] < TARGET_PER_CLASS or counts[2] < TARGET_PER_CLASS) and attempts < 200000:
        attempts += 1
        traj = random.choice(trajectories)
        if len(traj) < 2: continue
        
        t = random.randint(0, len(traj) - 2)
        item_t = traj[t]
        item_next = traj[t+1]
        
        # Check Timestamp continuity
        dt = item_next['timestamp'] - item_t['timestamp']
        if dt <= 0 or dt > 0.5: # Skip if duplicate or gap
            continue

        # Get Action
        l = item_t.get('left_cmd', 0)
        r = item_t.get('right_cmd', 0)
        action_id = discretize_action(l, r)
        
        # Filter: Only collect if we need more of this class
        if action_id not in [1, 2, 3]: continue
        if counts[action_id] >= TARGET_PER_CLASS: continue
            
        # Load Images
        path_t = os.path.join(DATA_ROOT, item_t['image_path'])
        path_next = os.path.join(DATA_ROOT, item_next['image_path'])
        
        if not os.path.exists(path_t) or not os.path.exists(path_next): continue

        img_t = load_image_gray(path_t)
        img_next = load_image_gray(path_next)
        
        if img_t is None or img_next is None: continue
        
        # Debug: Save first LEFT pair to verify visual movement
        if action_id == 1 and not saved_debug:
            cv2.imwrite("debug_left_t.jpg", img_t)
            cv2.imwrite("debug_left_next.jpg", img_next)
            logger.info(f"Saved debug images for Left action (dt={dt:.3f}s)")
            # Log diff
            diff = cv2.absdiff(img_t, img_next)
            cv2.imwrite("debug_left_diff.jpg", diff)
            non_zero = np.count_nonzero(diff)
            logger.info(f"Pixel Diff Count: {non_zero}")
            saved_debug = True
            
        # Compute Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            img_t, img_next, None, 
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Crop Center Strip (Middle 50% height)
        h_start = int(IMG_H * 0.25)
        h_end = int(IMG_H * 0.75)
        flow_crop = flow[h_start:h_end, :, :] 
        
        u = flow_crop[..., 0]
        mean_dx = np.mean(u)
        
        results.append({
            'action': action_id,
            'dx': mean_dx,
            'l': l,
            'r': r
        })
        counts[action_id] += 1
        
        if attempts % 1000 == 0:
            logger.info(f"Progress: {counts}")

    logger.info(f"Finished sampling: {counts}")
    if len(results) == 0:
        logger.error("No samples collected.")
        return

    # 3. Visualization
    logger.info("Generating plots...")
    
    actions = [r['action'] for r in results]
    dxs = [r['dx'] for r in results]
    
    # Map Action IDs to Names
    act_map = {1: "LEFT", 2: "RIGHT", 3: "STOP"}
    labels = [act_map[a] for a in actions]
    
    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=labels, y=dxs, order=["LEFT", "STOP", "RIGHT"], palette="muted")
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title("Horizontal Optical Flow per Action")
    plt.ylabel("Average Horizontal Pixel Shift (dx)")
    plt.xlabel("Action")
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    
    # Scatter Plot (dx vs Differential Torque)
    # Torque Diff = Right - Left. 
    # Left Turn (1): l=-80, r=-80. Diff = 0? Wait.
    # Discretize Action Logic:
    # Left (1): l < -1, r < -1. Both negative. Robot spins? 
    # If l=-80, r=-80. It's spinning CCW (if inverted wiring).
    # If standard wiring, (-,-) usually means BACK.
    # But user confirmed (-80, -80) is Left Turn.
    # Let's verify shift sign.
    # CCW Turn (Left) -> Camera pans Left -> Image shifts RIGHT (dx > 0).
    # CW Turn (Right) -> Camera pans Right -> Image shifts LEFT (dx < 0).
    
    # Plot dx vs Sample Index (sorted by action) is less useful.
    # Let's skip complex scatter for HTML report for now, just Violin is good.
    
    # 4. Generate HTML Report
    logger.info(f"Generating {OUTPUT_HTML}...")
    
    def get_stats(act_id):
        vals = [r['dx'] for r in results if r['action'] == act_id]
        if not vals: return 0, 0, 0
        return np.mean(vals), np.std(vals), len(vals)
        
    mu_L, std_L, n_L = get_stats(1)
    mu_R, std_R, n_R = get_stats(2)
    mu_S, std_S, n_S = get_stats(3)
    
    # Base64 Image
    def img_to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
            
    img_b64 = img_to_b64(OUTPUT_PLOT)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visual-Motor Alignment Check</title>
        <style>
             body {{ font-family: sans-serif; margin: 20px; background: #f8f9fa; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .stats-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            .stats-table th {{ background-color: #f2f2f2; }}
            .img-box {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
            .pass {{ color: green; font-weight: bold; }}
            .fail {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Visual-Motor Alignment Check (Optical Flow)</h1>
            <p>Analysis of {sum(counts.values())} balanced frame transitions $(I_t, I_{t+1})$ to verify that physical actions correspond to expected visual shifts.</p>
            
            <h3>Expected Behavior</h3>
            <ul>
                <li><strong>Left Turn (Action 1)</strong>: Turn CCW $\rightarrow$ Scene Shifts RIGHT ($dx > 0$)</li>
                <li><strong>Right Turn (Action 2)</strong>: Turn CW $\rightarrow$ Scene Shifts LEFT ($dx < 0$)</li>
                <li><strong>Stop (Action 3)</strong>: No Turn $\rightarrow$ Zero Shift ($dx \\approx 0$)</li>
            </ul>
            
            <h3>Results</h3>
            <table class="stats-table">
                <tr>
                    <th>Action</th>
                    <th>N Samples</th>
                    <th>Mean Shift (dx)</th>
                    <th>Std Dev</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td><strong>LEFT</strong></td>
                    <td>{n_L}</td>
                    <td>{mu_L:.2f} px</td>
                    <td>{std_L:.2f}</td>
                    <td class="{ 'pass' if mu_L > 0.5 else 'fail' }">{ "PASS" if mu_L > 0.5 else "FAIL" }</td>
                </tr>
                <tr>
                    <td><strong>RIGHT</strong></td>
                    <td>{n_R}</td>
                    <td>{mu_R:.2f} px</td>
                    <td>{std_R:.2f}</td>
                    <td class="{ 'pass' if mu_R < -0.5 else 'fail' }">{ "PASS" if mu_R < -0.5 else "FAIL" }</td>
                </tr>
                <tr>
                    <td><strong>STOP</strong></td>
                    <td>{n_S}</td>
                    <td>{mu_S:.2f} px</td>
                    <td>{std_S:.2f}</td>
                    <td class="{ 'pass' if abs(mu_S) < 0.5 else 'fail' }">{ "PASS" if abs(mu_S) < 0.5 else "FAIL" }</td>
                </tr>
            </table>
            
            <h3>Distribution Plot</h3>
            <div class="img-box">
                <img src="data:image/png;base64,{img_b64}" />
            </div>
            
            <p style="color: #777; font-size: 0.9em;">Generated by Embrained-AI</p>
        </div>
    </body>
    </html>
    """
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
        
    logger.info(f"Report saved to {OUTPUT_HTML}")
    print(f"Optical Flow Analysis Complete. Report: {os.path.abspath(OUTPUT_HTML)}")

if __name__ == "__main__":
    main()
