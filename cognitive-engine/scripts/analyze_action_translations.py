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
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def angular_distance(yaw1, yaw2):
    diff = (yaw2 - yaw1) % 360
    if diff > 180:
        diff -= 360
    return diff

def main():
    print("Loading telemetry...")
    df_tel = pd.read_csv("master_telemetry.csv")
    
    print("Loading transitions...")
    with open("data/all_transitions.json", "r") as f:
        transitions = json.load(f)
        
    print("Merging datasets...")
    # Convert 'ts' to string for robust matching, as floating point can jitter
    tel_map = {str(int(row['ts'])): row for _, row in df_tel.iterrows()}
    
    data = []
    
    # Action Map
    action_names = {1: 'FWD', 2: 'REV', 3: 'HW_LEFT', 4: 'HW_RIGHT', 5: 'SW_LEFT', 6: 'SW_RIGHT', 0: 'STOP'}
    
    for i in range(len(transitions) - 1):
        curr_t = transitions[i]
        next_t = transitions[i+1]
        
        # Ensure they are from the same session contiguous sequence
        if curr_t.get('session') != next_t.get('session'):
            continue
            
        action = curr_t.get('macro_action', 0)
        
        ts_curr = str(int(curr_t['timestamp'] * 1000))
        ts_next = str(int(next_t['timestamp'] * 1000))
        
        if ts_curr in tel_map and ts_next in tel_map:
            row_curr = tel_map[ts_curr]
            row_next = tel_map[ts_next]
            
            dx = row_next['cx'] - row_curr['cx']
            dy = row_next['cy'] - row_curr['cy']
            dist = np.sqrt(dx**2 + dy**2)
            
            dyaw = angular_distance(row_curr['yaw_deg'], row_next['yaw_deg'])
            
            data.append({
                'action_id': action,
                'action_name': action_names.get(action, f"ACT_{action}"),
                'dx': dx,
                'dy': dy,
                'dist_px': dist,
                'dyaw_deg': dyaw
            })

    if len(data) == 0:
        print("Error: No data matched!")
        return
        
    df_res = pd.DataFrame(data)
    print(f"Matched {len(df_res)} transitions with telemetry.")
    
    # Filter to main WASD actions
    df_res = df_res[df_res['action_id'].isin([1, 2, 3, 4])]
    
    summary = df_res.groupby('action_name').agg(
        count=('action_id', 'count'),
        avg_dist_px=('dist_px', 'mean'),
        std_dist_px=('dist_px', 'std'),
        avg_dyaw_deg=('dyaw_deg', 'mean'),
        std_dyaw_deg=('dyaw_deg', 'std')
    ).round(2)
    
    print("\n--- WASD Action Differentials ---")
    print(summary)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: XY Distance
    means_dist = summary['avg_dist_px']
    errs_dist = summary['std_dist_px']
    ax1.bar(means_dist.index, means_dist, yerr=errs_dist, capsize=5, color=['#4daf4a', '#e41a1c', '#377eb8', '#984ea3'])
    ax1.set_title("Average XY Translation per Action")
    ax1.set_ylabel("Distance (pixels)")
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Yaw Rotation
    means_yaw = summary['avg_dyaw_deg']
    errs_yaw = summary['std_dyaw_deg']
    ax2.bar(means_yaw.index, means_yaw, yerr=errs_yaw, capsize=5, color=['#4daf4a', '#e41a1c', '#377eb8', '#984ea3'])
    ax2.set_title("Average Yaw Rotation per Action")
    ax2.set_ylabel("Delta Yaw (degrees)\n(+) CCW/Left, (-) CW/Right")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_file = "action_differentials.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved visualization to {out_file}")

if __name__ == "__main__":
    main()
