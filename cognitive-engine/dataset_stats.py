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

import pandas as pd
import numpy as np
import glob
import os

dfs = []
for file in glob.glob('data/nook1d_*/episode_data.csv'):
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

if not dfs:
    print("No nook1d datasets found.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)

print(f"Dataset: nook1d (Aggregated {len(dfs)} folders)")
print(f"Total Transitions (Rows): {len(df)}")

print("\n--- Actions (pwm_left, pwm_right) ---")
if 'pwm_left' in df.columns and 'pwm_right' in df.columns:
    action_counts = df.groupby(['pwm_left', 'pwm_right']).size().reset_index(name='count')
    for index, row in action_counts.iterrows():
        print(f"Left: {row['pwm_left']:>4.0f}, Right: {row['pwm_right']:>4.0f} -> Count: {row['count']}")
else:
    print("pwm_left and pwm_right columns not found.")

if 'ir_reading' in df.columns:
    print("\n--- IR Reading Stats ---")
    print(f"Mean: {df['ir_reading'].mean():.2f}")
    print(f"Min:  {df['ir_reading'].min():.2f}")
    print(f"Max:  {df['ir_reading'].max():.2f}")
    print(f"Std:  {df['ir_reading'].std():.2f}")

if 'batt_raw' in df.columns:
    print("\n--- Battery Stats ---")
    print(f"Mean: {df['batt_raw'].mean():.2f}")
    print(f"Min:  {df['batt_raw'].min():.2f}")

if 'ping_raw' in df.columns:
    print("\n--- Ping Time Stats (ms) ---")
    print(f"Mean: {df['ping_raw'].mean():.2f}")
    print(f"Max:  {df['ping_raw'].max():.2f}")

if 'timestamp' in df.columns:
    print("\n--- Timestamp Stats ---")
    # Time between transitions (per file to avoid cross-file jumps)
    all_diffs = []
    for d in dfs:
        d = d.sort_values(by='timestamp')
        diffs = d['timestamp'].diff().dropna()
        all_diffs.append(diffs)
    
    diffs = pd.concat(all_diffs, ignore_index=True) if all_diffs else pd.Series(dtype=float)
    if not diffs.empty:
        print(f"Average time between transitions: {diffs.mean():.3f} seconds")
        print(f"Min time between transitions:     {diffs.min():.3f} seconds")
if 'timestamp' in df.columns and 'ping_raw' in df.columns:
    print("\n--- Correlation Stats ---")
    all_diffs = []
    all_pings = []
    for d in dfs:
        if 'ping_raw' not in d.columns:
            continue
        d = d.sort_values(by='timestamp')
        diffs = d['timestamp'].diff()
        valid_idx = diffs.notna()
        all_diffs.append(diffs[valid_idx])
        all_pings.append(d['ping_raw'][valid_idx])
        
    if all_diffs:
        diff_series = pd.concat(all_diffs, ignore_index=True)
        ping_series = pd.concat(all_pings, ignore_index=True)
        correlation = ping_series.corr(diff_series)
        print(f"Pearson Correlation (Ping Time vs Time Between Transitions): {correlation:.4f}")
