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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import glob as glob_module

def load_data():
    base_dir = r"c:\Users\chris\Embrained\software_suite\data"
    dirs = glob_module.glob(os.path.join(base_dir, "markov_*"))
    
    new_dates = ["2026-04-09_17-57-21", "2026-04-09_18-08-14"]
    
    all_rows = []
    
    try:
        master_telemetry = pd.read_csv(r"c:\Users\chris\Embrained\software_suite\master_telemetry.csv")
        # group by directory
        grouped = master_telemetry.groupby('img_dir')
        master_dict = {}
        for d, df in grouped:
            df = df.sort_values('ts')
            df['next_cx'] = df['cx'].shift(-1)
            df['next_cy'] = df['cy'].shift(-1)
            df['next_yaw'] = df['yaw_deg'].shift(-1)
            
            # calculate deltas
            df['delta_pixels'] = np.sqrt((df['next_cx'] - df['cx'])**2 + (df['next_cy'] - df['cy'])**2)
            
            # calculate delta yaw strictly bounded between -180 and 180
            dy = df['next_yaw'] - df['yaw_deg']
            dy = (dy + 180) % 360 - 180
            df['delta_yaw'] = dy
            
            master_dict[d] = df.set_index('ts')
    except Exception as e:
        print("Could not load master_telemetry.csv:", e)
        master_dict = {}

    for d in dirs:
        d_name = os.path.basename(d)
        is_new = any(n in d_name for n in new_dates)
        group_label = "New" if is_new else "Original"
        
        episode_csv = os.path.join(d, "episode_data.csv")
        
        if os.path.exists(episode_csv):
            df_merged = pd.read_csv(episode_csv)
            # rename timestamp to ts for joining
            df_merged.rename(columns={'timestamp': 'ts'}, inplace=True)
            
            # Extract action from PWM
            def get_action_id(row):
                l, r = row['pwm_left'], row['pwm_right']
                if l == 0 and r == 0: return 5 # Or 0 depending on stop vs int_stop
                if l > 0 and r > 0: return 1
                if l < 0 and r < 0: return 2
                if l < 0 and r > 0: return 3
                if l > 0 and r < 0: return 4
                return 0
                
            df_merged['action'] = df_merged.apply(get_action_id, axis=1)
            
            # Map ir_reading to ir
            df_merged.rename(columns={'ir_reading': 'ir'}, inplace=True)
            
            # Add geometric metrics if available
            img_dir_path = os.path.normpath(os.path.join(d, "images")).lower()
            
            # Find matching key in master_dict
            match_key = None
            for k in master_dict.keys():
                if os.path.normpath(k).lower() == img_dir_path:
                    match_key = k
                    break
                    
            if match_key:
                m_df = master_dict[match_key]
                m_df.index = m_df.index.astype(float)
                df_merged['ts'] = (df_merged['ts'].astype(float) * 1000).round()
                # align by ts using nearest to avoid float mismatch
                df_merged = pd.merge_asof(
                    df_merged.sort_values('ts'), 
                    m_df[['delta_pixels', 'delta_yaw']].sort_index(), 
                    left_on='ts', 
                    right_index=True, 
                    direction='nearest', 
                    tolerance=50.0
                )
                print(f"Merged telemetry for {d_name}: {len(m_df)} rows")
            else:
                df_merged['delta_pixels'] = np.nan
                df_merged['delta_yaw'] = np.nan
                print(f"Skipped telemetry for {d_name}")
                
            df_merged['group'] = group_label
            df_merged['dataset'] = d_name
            all_rows.append(df_merged)
            
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

def plot_distributions(df):
    output_dir = r"c:\Users\chris\.gemini\antigravity\brain\a2b2ffb4-73e1-47c2-bdae-eb297e9b4466"
    
    # Map action IDs
    action_map = {0: 'STOP', 1: 'FWD', 2: 'BACK', 3: 'LEFT', 4: 'RIGHT', 5: 'INT_STOP'}
    df['action_name'] = df['action'].map(action_map).fillna('UNKNOWN')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Action Counts
    plt.figure(figsize=(10, 6))
    counts = df.groupby(['group', 'action_name']).size().reset_index(name='count')
    # Normalize by number of runs to get average actions per run
    n_orig = df[df['group'] == 'Original']['dataset'].nunique()
    n_new = df[df['group'] == 'New']['dataset'].nunique()
    
    counts['norm_count'] = counts.apply(lambda x: x['count'] / (n_orig if x['group'] == 'Original' else n_new), axis=1)
    
    sns.barplot(data=counts, x='action_name', y='norm_count', hue='group')
    plt.title("Expected Actions per Recording Session")
    plt.ylabel("Average Count per Session")
    plt.savefig(os.path.join(output_dir, "analysis_acts.png"))
    plt.close()
    
    # 2. IR Distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='group', y='ir')
    plt.title("IR Sensor Distribution")
    plt.savefig(os.path.join(output_dir, "analysis_ir.png"))
    plt.close()
    
    # Filter moving actions
    df_moves = df[df['action'].isin([1, 2, 3, 4])]
    
    # 3. Pixel Displacement
    plt.figure(figsize=(12, 6))
    df_plot1 = df_moves.dropna(subset=['delta_pixels'])
    if not df_plot1.empty:
        sns.boxplot(data=df_plot1, x='action_name', y='delta_pixels', hue='group')
        plt.title("Geometric Pixel Displacement by Action Type")
        plt.savefig(os.path.join(output_dir, "analysis_disp.png"))
    plt.close()
    
    # 4. Yaw Delta
    plt.figure(figsize=(12, 6))
    df_plot2 = df_moves.dropna(subset=['delta_yaw'])
    if not df_plot2.empty:
        sns.boxplot(data=df_plot2, x='action_name', y='delta_yaw', hue='group')
        plt.title("Geometric Yaw Delta (degrees) by Action Type")
        plt.savefig(os.path.join(output_dir, "analysis_yaw.png"))
    plt.close()

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    if not df.empty:
        print("Data loaded. Total transitions:", len(df))
        print("Generating plots...")
        plot_distributions(df)
        print("Generated 4 graphs!")
    else:
        print("No data found.")
