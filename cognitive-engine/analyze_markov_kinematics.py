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
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.extract_telemetry import TelemetryExtractor

def get_action_id(left, right):
    if left == 0 and right == 0: return 0 # Stop
    if left > 0 and right > 0: return 1   # Forward
    if left < 0 and right < 0: return 2   # Reverse
    if left > 0 and right < 0: return 3   # Right Turn
    if left < 0 and right > 0: return 4   # Left Turn
    return -1

def run_analysis():
    print("Loading Extractor...")
    extractor = TelemetryExtractor([])
    extractor.load_cache('data/telemetry_cache.npz')
    extractor._precompute_rotations()
    
    markov_dirs = glob.glob('data/markov_*')
    
    actions_data = {i: {'dx': [], 'dy': [], 'dist': [], 'dtheta': []} for i in range(5)}
    
    for d in markov_dirs:
        print(f"Processing dataset: {d}")
        csv_path = os.path.join(d, 'episode_data.csv')
        df = pd.read_csv(csv_path)
        
        last_feats = None
        last_action = None
        
        # Reset extractor state for new continuous dataset
        extractor.last_cx = None
        extractor.last_cy = None
        extractor.last_theta = None
        extractor.last_feats = None
        extractor.last_seen_time = 0
        
        for idx, row in df.iterrows():
            img_file = row['image_file']
            if not img_file.startswith('webcam_'):
                img_file = 'webcam_' + img_file
            img_path = os.path.join(d, 'images', img_file)
            if not os.path.exists(img_path): continue
            
            img_bgr = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # The telemetry pipeline resizes images normally! But markov records full sizes or resized?
            # Wait, markov recordings are usually 320x240? Or 640x480?
            if img_gray.shape != (480, 640):
                img_gray = cv2.resize(img_gray, (640, 480))
            
            feats = extractor.process_single_frame(img_gray)
            
            action_id = get_action_id(row['pwm_left'], row['pwm_right'])
            
            if last_feats is not None and feats is not None and last_action != -1:
                cx1, cy1, th1 = last_feats['raw_cx'], last_feats['raw_cy'], last_feats['best_theta']
                cx2, cy2, th2 = feats['raw_cx'], feats['raw_cy'], feats['best_theta']
                
                # We need to compute dx, dy in the LOCAL frame of the robot!
                # Ah, the user didn't specify local, but "change in XY"
                dx_global = cx2 - cx1
                dy_global = cy2 - cy1
                dist = np.sqrt(dx_global**2 + dy_global**2)
                
                # Change in orientation
                th_diff = th2 - th1
                # Normalize to -180 to 180
                th_diff = (th_diff + 180) % 360 - 180
                
                # To be physically meaningful, let's just record raw dist and dtheta for now
                if dist < 100:  # Ignore massive glitches/teleports
                    actions_data[last_action]['dist'].append(dist)
                    actions_data[last_action]['dtheta'].append(th_diff)
            
            last_feats = feats
            last_action = action_id

    # Plotting
    action_names = {0: "Stop", 1: "Forward", 2: "Reverse", 3: "Right", 4: "Left"}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Kinematic Impact of Discrete Actions (Extracted via Oracle Vision)", fontsize=16)
    
    avg_dists = [np.mean(actions_data[i]['dist']) if actions_data[i]['dist'] else 0 for i in range(5)]
    std_dists = [np.std(actions_data[i]['dist']) if actions_data[i]['dist'] else 0 for i in range(5)]
    
    avg_thetas = [np.mean(actions_data[i]['dtheta']) if actions_data[i]['dtheta'] else 0 for i in range(5)]
    std_thetas = [np.std(actions_data[i]['dtheta']) if actions_data[i]['dtheta'] else 0 for i in range(5)]
    
    axes[0].bar([action_names[i] for i in range(5)], avg_dists, yerr=std_dists, capsize=5, color='royalblue')
    axes[0].set_title("Average Distance Traveled Per Tick (Pixels)")
    axes[0].set_ylabel("Pixels")
    
    axes[1].bar([action_names[i] for i in range(5)], avg_thetas, yerr=std_thetas, capsize=5, color='darkorange')
    axes[1].set_title("Average Orientation Change Per Tick (Degrees)")
    axes[1].set_ylabel("Degrees")
    axes[1].axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('kinematic_analysis.png', dpi=150)
    print("Saved plot to kinematic_analysis.png")

if __name__ == "__main__":
    run_analysis()
