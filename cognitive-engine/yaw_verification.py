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
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from natsort import natsorted

# CONFIGURATION
DATASET_DIR = os.path.join("data", "vint_formatted_livingroom")
OUTPUT_PLOT = "yaw_verification.png"

def load_trajectory(traj_folder):
    # Load Poses
    pkl_path = os.path.join(traj_folder, "traj_data.pkl")
    if not os.path.exists(pkl_path): return None
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Load Images
    image_paths = natsorted(glob(os.path.join(traj_folder, "*.jpg")))
    
    return data["position"], data["yaw"], image_paths

def find_clean_turn(dataset_dir):
    """
    Scans trajectories to find a sequence of 5 frames where:
    1. The robot is turning continuously (Yaw changes > 0.1 rad).
    2. The direction is consistent (all Positive or all Negative).
    """
    traj_folders = natsorted(glob(os.path.join(dataset_dir, "trajectory_*")))
    
    print(f"Scanning {len(traj_folders)} trajectories for a clean turn...")
    
    for folder in traj_folders:
        res = load_trajectory(folder)
        if not res: continue
        pos, yaw, imgs = res
        
        # Calculate Delta Yaw (Angular Velocity)
        # We need a window of 5 frames
        for i in range(len(yaw) - 6):
            window_yaw = yaw[i:i+6]
            deltas = np.diff(window_yaw)
            
            # Check 1: Significant Motion (Absolute delta > 0.05 rad per step)
            if not all(abs(d) > 0.05 for d in deltas): continue
            
            # Check 2: Consistent Direction (All positive or all negative)
            if all(d > 0 for d in deltas):
                return folder, i, "Left Turn (Positive Yaw)"
            elif all(d < 0 for d in deltas):
                return folder, i, "Right Turn (Negative Yaw)"
                
    return None, None, None

def visualize_sequence(folder, start_idx, turn_type):
    pos, yaw, img_paths = load_trajectory(folder)
    
    # Select 5 frames
    indices = range(start_idx, start_idx + 5)
    
    plt.figure(figsize=(20, 6))
    plt.suptitle(f"Yaw Verification: {turn_type}\nSource: {folder}", fontsize=16)
    
    for i, idx in enumerate(indices):
        # Load Image
        img = cv2.imread(img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Current Heading
        current_yaw = yaw[idx]
        delta_yaw = yaw[idx] - yaw[start_idx] # Relative to start
        
        ax = plt.subplot(1, 5, i+1)
        ax.imshow(img)
        ax.set_title(f"Frame {idx}\nYaw: {np.degrees(delta_yaw):.1f}°")
        ax.axis('off')
        
        # Draw Arrow indicating "Where the Odometry thinks we are going"
        # If Yaw is Positive (Left), Arrow points Left.
        # Length of arrow proportional to turn sharpness.
        arrow_len = 0.4
        dx = -np.sin(delta_yaw) * arrow_len # visual x is opposite to math angle
        dy = 0 # simple 1D visual
        
        # Note: This isn't a map, just a sanity check text.
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"[*] Generated verification plot: {OUTPUT_PLOT}")
    print(f"[*] Inspect this image. If the images rotate AGAINST the labels, Yaw is Inverted.")

if __name__ == "__main__":
    folder, idx, turn_type = find_clean_turn(DATASET_DIR)
    
    if folder:
        visualize_sequence(folder, idx, turn_type)
    else:
        print("[!] No suitable clean turns found. Try checking the raw data or lowering thresholds.")