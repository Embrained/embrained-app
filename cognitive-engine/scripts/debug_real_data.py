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

import csv
import re
import os
import sys

# Add parent directory to path to allow imports from backend/modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def analyze_consecutive_stops(data_dir):
    motor_re = re.compile(r'l: *(-?\d+);r: *(-?\d+);')
    
    datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("capture")]
    if not datasets:
        print("No datasets found")
        return

    # Analyze first dataset
    ds_path = os.path.join(data_dir, datasets[0], "log.csv")
    print(f"Analyzing {ds_path}...")
    
    cmds = []
    try:
        with open(ds_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                m_cmd_str = row.get('motor_cmd', '')
                m = motor_re.search(m_cmd_str)
                if m:
                    l_cmd, r_cmd = map(int, m.groups())
                    cmds.append((l_cmd, r_cmd))
                else:
                    cmds.append((0,0))
    except Exception as e:
        print(f"Error: {e}")
        return

    stable_stops = 0
    total_stops = 0
    
    for i in range(1, len(cmds)):
        curr = cmds[i]
        prev = cmds[i-1]
        
        if curr == (0,0):
            total_stops += 1
            if prev == (0,0):
                stable_stops += 1

    print(f"Total frames: {len(cmds)}")
    print(f"Total 'stop' frames (0,0): {total_stops}")
    print(f"Stable stop events (consecutive 0,0): {stable_stops}")
    
    if stable_stops == 0 and total_stops > 0:
        print("\nCONCLUSION: Stops exist but are never consecutive. Current logic requires 2 consecutive stops.")
    elif stable_stops > 0:
        print("\nCONCLUSION: Stable stops exist. The issue might be segment length.")

if __name__ == "__main__":
    analyze_consecutive_stops("data")
