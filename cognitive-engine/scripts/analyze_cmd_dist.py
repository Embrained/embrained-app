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


import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIG ---
DATA_PATH = "data/nook/all_transitions.json"
OUTPUT_PLOT = "cmd_distribution.png"

def analyze_commands():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Loading transitions...")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    left_cmds = []
    right_cmds = []
    
    for item in data:
        # Check if cmd is float or int
        l = float(item.get('left_cmd', 0))
        r = float(item.get('right_cmd', 0))
        left_cmds.append(l)
        right_cmds.append(r)
        
    left_cmds = np.array(left_cmds)
    
    print(f"Loaded {len(left_cmds)} samples.")
    print(f"Left Cmd Range: [{left_cmds.min()}, {left_cmds.max()}]")
    
    # Plot Histogram
    plt.figure(figsize=(10, 6))
    # Use log scale for y to see rare high-value commands if they are sparse
    plt.hist(left_cmds, bins=100, range=(-100, 100), log=True, color='blue', alpha=0.7, label='Left Cmd')
    plt.title("Distribution of Motor Commands (Log Scale)")
    plt.xlabel("Command Value")
    plt.ylabel("Count (Log)")
    plt.axvline(x=40, color='r', linestyle='--', label='Threshold 40')
    plt.axvline(x=-40, color='r', linestyle='--')
    plt.axvline(x=1, color='g', linestyle='--', label='Threshold 1')
    plt.axvline(x=-1, color='g', linestyle='--')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig(OUTPUT_PLOT)
    print(f"Histogram saved to {OUTPUT_PLOT}")
    
    # Calculate stats above/below thresholds
    count_active_1 = np.sum(np.abs(left_cmds) > 1)
    count_active_40 = np.sum(np.abs(left_cmds) > 40)
    
    print(f"Count |cmd| > 1:  {count_active_1} ({count_active_1/len(left_cmds)*100:.2f}%)")
    print(f"Count |cmd| > 40: {count_active_40} ({count_active_40/len(left_cmds)*100:.2f}%)")

if __name__ == "__main__":
    analyze_commands()
