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
import csv
import re
import glob
import sys

# Add parent directory to path to allow imports from backend/modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def discretize_action(left, right):
    """
    Maps motor commands to Action IDs.
    MUST MATCH backend/train_cql.py logic!
    """
    tol = 1 # NEW TOLERANCE
    
    # STOP (3)
    if abs(left) < 1 and abs(right) < 1: 
        return 3
        
    # FWD (0) -> l:-S, r:S
    if left < -tol and right > tol: 
        return 0
        
    # LEFT (1) -> l:-S, r:-S
    if left < -tol and right < -tol: 
        return 1
        
    # RIGHT (2) -> l:S, r:S
    if left > tol and right > tol: 
        return 2
        
    # BACK (4) -> l:S, r:-S
    if left > tol and right < -tol: 
        return 4
        
    return 3 # Default to STOP if ambiguous

def analyze_balance(data_dir):
    motor_re = re.compile(r'l: *(-?\d+);r: *(-?\d+);')
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_samples = 0
    
    # Find all log.csv files
    files = glob.glob(os.path.join(data_dir, "**", "log.csv"), recursive=True)
    print(f"Found {len(files)} log files.")
    
    for log_path in files:
        # print(f"Processing {log_path}...")
        try:
            with open(log_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    m_cmd_str = row.get('motor_cmd', '')
                    m = motor_re.search(m_cmd_str)
                    if m:
                        ls = int(m.group(1))
                        rs = int(m.group(2))
                        action = discretize_action(ls, rs)
                        action_counts[action] += 1
                        total_samples += 1
        except Exception as e:
            print(f"Error reading {log_path}: {e}")
            
    print("-" * 30)
    print(f"TOTAL SAMPLES: {total_samples}")
    print("-" * 30)
    
    action_names = {0: "FWD", 1: "LEFT", 2: "RIGHT", 3: "STOP", 4: "BACK"}
    
    for k in sorted(action_counts.keys()):
        count = action_counts[k]
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{action_names[k]:<6} (ID {k}): {count} ({pct:.1f}%)")
        
if __name__ == "__main__":
    analyze_balance("data")
