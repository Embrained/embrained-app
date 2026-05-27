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
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ActionLagCheck")

# Configuration
DATASET_NAME = "nook"
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", DATASET_NAME)
TRANSITIONS_PATH = os.path.join(DATA_ROOT, 'all_transitions.json')
OUTPUT_PLOT = "action_lag_plot.png"

IMG_H = 64
IMG_W = 64

def discretize_action(left, right):
    """
    Maps motor commands to Action IDs.
    Phys FWD  (0) <- l:-S, r:S
    Phys LEFT (1) <- l:-S, r:-S
    Phys RIGHT(2) <- l:S, r:S
    Phys STOP (3) <- l:0, r:0
    Phys BACK (4) <- l:S, r:-S
    """
    tol = 40
    if abs(left) < 1 and abs(right) < 1: return 3 # STOP
    if left < -tol and right > tol: return 0 # FWD
    if left < -tol and right < -tol: return 1 # LEFT
    if left > tol and right > tol: return 2 # RIGHT
    if left > tol and right < -tol: return 4 # BACK
    return 3

def load_image_gray(path):
    if not os.path.exists(path): return None
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_W, IMG_H))
    return img

def save_debug_sequence(imgs, action_name, idx):
    """Saves a sequence of images to lag_debug/ folder."""
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lag_debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save montage: horizontal strip of all images
    # Downsample for size
    display_imgs = []
    for i, img in enumerate(imgs):
        # Annotate
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color = (0, 255, 0) if i >= 10 else (0, 0, 255) # Green for Move, Red for Pre-Move
        cv2.putText(disp, str(i-10), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        display_imgs.append(disp)
        
    montage = np.hstack(display_imgs)
    cv2.imwrite(os.path.join(debug_dir, f"trigger_{action_name}_{idx}.jpg"), montage)

def main():
    logger.info("Starting Action-Triggered Visual Flow Analysis (Lag Check)...")
    
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

    # 2. Identify Triggers (Stop -> Left/Right)
    WINDOW = 10 # +/- 10 frames
    
    left_epochs = []
    right_epochs = []
    
    debug_counts = {"left": 0, "right": 0}

    logger.info("Extracting epochs around Action Onset...")
    
    for traj in trajectories:
        if len(traj) < WINDOW * 2: continue
        
        # Pre-calculate actions for speed
        actions = []
        for item in traj:
            actions.append(discretize_action(item.get('left_cmd',0), item.get('right_cmd',0)))
            
        for i in range(WINDOW, len(traj) - WINDOW - 1):
            prev_act = actions[i-1]
            curr_act = actions[i]
            
            # Detect Onset
            is_onset = (prev_act == 3) and (curr_act != 3)
            if not is_onset: continue
            
            # Check Trigger Type
            target_list = None
            action_name = ""
            if curr_act == 1: 
                target_list = left_epochs
                action_name = "left"
            elif curr_act == 2: 
                target_list = right_epochs
                action_name = "right"
            else: continue
            
            # Extract Epoch (Time and Images)
            # We need flow from t-WINDOW to t+WINDOW
            # Flow at k is diff(img[k], img[k+1])
            epoch_flows = []
            valid_epoch = True
            
            start_idx = i - WINDOW
            end_idx = i + WINDOW
            
            # Load images for the window can be slow, but we only do it for triggers
            # Optimization: Load images on demand
            
            imgs = []
            for k in range(start_idx, end_idx + 1): # Need +1 for flow
                path = os.path.join(DATA_ROOT, traj[k]['image_path'])
                img = load_image_gray(path)
                if img is None: 
                    valid_epoch = False
                    break
                imgs.append(img)
                
            if not valid_epoch: continue
            
            # Debug: Save first 3 examples of each action
            if debug_counts[action_name] < 3:
                save_debug_sequence(imgs, action_name, debug_counts[action_name])
                debug_counts[action_name] += 1
            
            # Calculate Flow Trace
            for k in range(len(imgs) - 1):
                # Larger winsize=30 for bigger/faster motions
                flow = cv2.calcOpticalFlowFarneback(
                    imgs[k], imgs[k+1], None, 
                    pyr_scale=0.5, levels=3, winsize=30, iterations=3, poly_n=7, poly_sigma=1.5, flags=0
                )
                
                # Center Strip
                h_start = int(IMG_H * 0.25)
                h_end = int(IMG_H * 0.75)
                flow_crop = flow[h_start:h_end, :, :] 
                u = flow_crop[..., 0]
                epoch_flows.append(np.mean(u))
                
            target_list.append(epoch_flows)
            
            if len(left_epochs) + len(right_epochs) > 500: break # Limit samples
        
        if len(left_epochs) + len(right_epochs) > 500: break

    logger.info(f"Collected {len(left_epochs)} Left Onsets and {len(right_epochs)} Right Onsets.")
    
    if not left_epochs and not right_epochs:
        logger.error("No valid onsets found.")
        return

    # 3. Aggregate & Plot
    x_axis = np.arange(-WINDOW, WINDOW) # 20 points. Flow has 20 points (imgs 21)
    
    plt.figure(figsize=(12, 6))
    
    if left_epochs:
        mean_left = np.mean(left_epochs, axis=0) # Shape (20,)
        std_left = np.std(left_epochs, axis=0)
        plt.plot(x_axis, mean_left, label=f'Left Turn Onset (N={len(left_epochs)})', color='blue', linewidth=2)
        plt.fill_between(x_axis, mean_left - std_left*0.2, mean_left + std_left*0.2, color='blue', alpha=0.1)
        
    if right_epochs:
        mean_right = np.mean(right_epochs, axis=0)
        std_right = np.std(right_epochs, axis=0)
        plt.plot(x_axis, mean_right, label=f'Right Turn Onset (N={len(right_epochs)})', color='red', linewidth=2)
        plt.fill_between(x_axis, mean_right - std_right*0.2, mean_right + std_right*0.2, color='red', alpha=0.1)
        
    plt.axvline(0, color='black', linestyle='--', label='Action Onset (t=0)')
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    plt.title("Action-Triggered Visual Flow Response (ERP Analysis)")
    plt.xlabel("Time Relative to Action Onset (Frames)")
    plt.ylabel("Mean Horizontal Flow (dx)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_PLOT)
    logger.info(f"Plot saved to {OUTPUT_PLOT}")
    print(f"Action Lag Analysis Complete. Plot: {os.path.abspath(OUTPUT_PLOT)}")

if __name__ == "__main__":
    main()
