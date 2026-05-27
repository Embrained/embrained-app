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
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_telemetry_plots(data_root, prefix_name):
    # Setup paths
    transitions_path = os.path.join(data_root, "all_transitions.json")
    telemetry_path = os.path.join(data_root, "master_telemetry.csv")
    
    if not os.path.exists(transitions_path) or not os.path.exists(telemetry_path):
        logger.warning(f"Skipping visualization, missing data at {data_root}")
        return False
        
    try:
        with open(transitions_path, "r") as f:
            transitions = json.load(f)
            
        df_telemetry = pd.read_csv(telemetry_path)
        
        # Identity colors
        color_blue = '#4169e1'
        color_purple = '#8a2be2'
        color_green = '#3cb44b'
        color_orange = '#ff8c00'
        color_yellow = '#ffd700'
        color_grey = '#333333'
        
        color_map = {
            0: color_grey,    # STOP
            1: '#4caf50',     # FWD (Green)
            2: '#9c27b0',     # REV (Purple)
            3: '#e51c23',     # HW_LEFT (Red)
            4: '#2582bd',     # HW_RIGHT (Blue)
            5: color_green    # INT STOP
        }

        action_labels = {
            0: 'STOP', 1: 'FWD', 2: 'REV', 3: 'HW_LEFT',  4: 'HW_RIGHT', 5: 'INT_STOP'
        }
        
        for t in transitions:
            img_path = t.get("image_path", "")
            ts_str = img_path.split("frame_")[-1].replace(".jpg", "")
            if ts_str.isdigit():
                t["ts"] = int(ts_str)
            else:
                t["ts"] = 0
                
        df_trans = pd.DataFrame(transitions)
        df_trans['ts'] = df_trans['ts'].astype(str)
        df_telemetry['ts'] = df_telemetry['ts'].astype(str)
        
        df = pd.merge(df_trans, df_telemetry, on='ts', how='inner')
        if len(df) == 0:
            logger.warning("No matches between transitions and telemetry!")
            return False
            
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Normalize bounds per session for spatial mapping
        df['nx'] = 0.0
        df['ny'] = 0.0
        for session, grp in df.groupby('session'):
            min_x, max_x = grp['cx'].min(), grp['cx'].max()
            min_y, max_y = grp['cy'].min(), grp['cy'].max()
            if max_x - min_x > 0 and max_y - min_y > 0:
                df.loc[grp.index, 'nx'] = (grp['cx'] - min_x) / (max_x - min_x)
                df.loc[grp.index, 'ny'] = (grp['cy'] - min_y) / (max_y - min_y)

        # Plot 1: Spatial Map
        fig, ax = plt.subplots(figsize=(10, 8))
        for action_id, grp in df.groupby('macro_action'):
            ax.scatter(grp['nx'], grp['ny'], c=color_map.get(action_id, color_grey), 
                       label=action_labels.get(action_id, 'UNKNOWN'), alpha=0.5, s=15, edgecolors='none')
        ax.set_title("Allocentric Telemetry & Action Mapping", fontsize=16, fontweight='bold', color=color_grey)
        ax.set_xlabel("Normalized X Coordinate [0,1]", fontsize=12, color=color_grey)
        ax.set_ylabel("Normalized Y Coordinate [0,1]", fontsize=12, color=color_grey)
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_root, f"{prefix_name}_action_spatial_map.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Action Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = df['macro_action'].value_counts()
        labels = [action_labels.get(i, str(i)) for i in counts.index]
        colors = [color_map.get(i, color_grey) for i in counts.index]
        ax.bar(labels, counts.values, color=colors, alpha=0.8)
        ax.set_title("Macro Action Distribution", fontsize=16, fontweight='bold', color=color_grey)
        ax.set_ylabel("Count", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(data_root, f"{prefix_name}_action_distribution.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
        # Plot 3: IR Distribution
        if 'ir' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=df, x='ir', kde=True, color='#ff8c00', fill=True, ax=ax)
            ax.set_title("Hardware IR Sensor Density Distribution", fontsize=16, fontweight='bold', color=color_grey)
            ax.set_xlabel("Raw Sharp IR Sensor Signal (Higher = Closer)", fontsize=12)
            ax.set_ylabel("Density / Frequency", fontsize=12)
            plt.axvline(x=1500, color='red', linestyle='--', linewidth=2, label="IR Reflex Threshold (1500)")
            ax.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(data_root, f"{prefix_name}_ir_distribution.png"), dpi=200, bbox_inches='tight')
            plt.close()

        # Plot 4: Action Effect Deltas
        df_sorted = df.sort_values(by=['session', 'ts']).reset_index(drop=True)
        df_sorted['next_session'] = df_sorted['session'].shift(-1)
        df_sorted['next_cx'] = df_sorted['cx'].shift(-1)
        df_sorted['next_cy'] = df_sorted['cy'].shift(-1)
        df_sorted['next_yaw'] = df_sorted['yaw_deg'].shift(-1)
        
        valid_mask = df_sorted['session'] == df_sorted['next_session']
        df_valid = df_sorted[valid_mask].copy()
        
        df_valid['delta_dist'] = np.sqrt((df_valid['next_cx'] - df_valid['cx'])**2 + (df_valid['next_cy'] - df_valid['cy'])**2)
        
        dyw = df_valid['next_yaw'] - df_valid['yaw_deg']
        dyw = (dyw + 180) % 360 - 180
        df_valid['delta_yaw'] = dyw
        
        target_actions = [1, 3, 4, 2]
        ta_labels = ['FWD', 'HW_LEFT', 'HW_RIGHT', 'REV']
        ta_colors = [color_map[a] for a in target_actions]
        
        dist_means = []
        dist_stds = []
        yaw_means = []
        yaw_stds = []
        
        for a in target_actions:
            sub = df_valid[df_valid['macro_action'] == a]
            if len(sub) > 0:
                dist_means.append(sub['delta_dist'].mean())
                dist_stds.append(sub['delta_dist'].std())
                yaw_means.append(sub['delta_yaw'].mean())
                yaw_stds.append(sub['delta_yaw'].std())
            else:
                dist_means.append(0)
                dist_stds.append(0)
                yaw_means.append(0)
                yaw_stds.append(0)
                
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(ta_labels, dist_means, yerr=dist_stds, capsize=5, color=ta_colors, alpha=0.9)
        axes[0].set_title("Average XY Translation per Action", fontsize=14, color=color_grey)
        axes[0].set_ylabel("Distance (pixels)")
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(ta_labels, yaw_means, yerr=yaw_stds, capsize=5, color=ta_colors, alpha=0.9)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_title("Average Yaw Rotation per Action", fontsize=14, color=color_grey)
        axes[1].set_ylabel("Delta Yaw (degrees)\n(+) CCW/Left, (-) CW/Right")
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_root, f"{prefix_name}_action_telemetry_effect.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualized telemetry analytics to {data_root} with prefix {prefix_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to assemble visual telemetry analytics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    generate_telemetry_plots("data", "20260410_test")
