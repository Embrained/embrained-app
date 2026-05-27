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
import shutil
import pickle
import csv
import re
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_DATASET = os.path.join("data", "livingroom")
OUTPUT_DATASET = os.path.join("data", "vint_formatted_livingroom")

# CALIBRATION RESULTS (Hybrid Method)
# 1. Rotational: From Nelder-Mead Optimization (Accounts for carpet slip)
WHEELBASE = 0.173       

# 2. Geometric: From Bounding Box (26m -> 4m)
VELOCITY_SCALE = 0.15   

# Frame Timing
DT = 0.1                
LAG_FRAMES = 2          

# ViNT Specs
METRIC_WAYPOINT_SPACING = 0.3 
# =================================================

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def parse_motor_cmd(cmd_str):
    if not isinstance(cmd_str, str): return 0, 0
    match = re.search(r'l:(-?\d+);r:(-?\d+);', cmd_str)
    if match: return int(match.group(1)), int(match.group(2))
    return 0, 0

def pwm_to_velocity(pwm, is_left_motor=False):
    """
    FINAL HYBRID MAPPING:
    Uses base speeds scaled by the Geometric Factor (0.15).
    """
    if abs(pwm) < 40: return 0.0

    mag = abs(pwm)
    
    # Base speeds (Raw relative difference)
    if mag <= 100:   base = 0.05
    elif mag <= 200: base = 0.09
    else:            base = 0.12
    
    # Apply Geometric Scale (0.15)
    speed = base * VELOCITY_SCALE 
    
    velocity = speed if pwm > 0 else -speed
    if is_left_motor: velocity = -velocity

    return velocity

def generate_trajectory(source_folder, dest_folder, traj_id):
    images = natsorted(glob(os.path.join(source_folder, "*.jpg")))
    log_path = os.path.join(source_folder, "log.csv")
    
    if not images or not os.path.exists(log_path): return False

    raw_commands = []
    try:
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
            for row in reader:
                l, r = parse_motor_cmd(row.get('motor_cmd', ''))
                raw_commands.append((l, r))
    except: return False

    if LAG_FRAMES > 0:
        aligned_commands = [(0,0)] * LAG_FRAMES + raw_commands[:-LAG_FRAMES]
    else:
        aligned_commands = raw_commands

    length = min(len(images), len(aligned_commands))
    
    temp_pos = []
    temp_yaw = []
    is_moving_mask = []
    
    x, y, theta = 0.0, 0.0, 0.0
    
    for t in range(length):
        l_pwm, r_pwm = aligned_commands[t]
        v_l = pwm_to_velocity(l_pwm, is_left_motor=True)
        v_r = pwm_to_velocity(r_pwm, is_left_motor=False)
        
        # Movement Check (Threshold scaled by 0.15)
        is_moving = abs(v_l) > (0.001 * VELOCITY_SCALE) or abs(v_r) > (0.001 * VELOCITY_SCALE)
        is_moving_mask.append(is_moving)

        v_robot = (v_l + v_r) / 2.0
        w_robot = (v_r - v_l) / WHEELBASE
        
        # Standard Kinematics
        x += v_robot * np.cos(theta) * DT
        y += v_robot * np.sin(theta) * DT
        theta += w_robot * DT
        
        temp_pos.append([x, y])
        temp_yaw.append(theta)

    # Smart Trim
    try:
        true_indices = [i for i, x in enumerate(is_moving_mask) if x]
        if not true_indices: return False
        start_idx = max(0, true_indices[0] - 5)
        end_idx = min(length, true_indices[-1] + 5)
    except: return False
    
    if (end_idx - start_idx) < 20: return False

    final_pos = np.array(temp_pos[start_idx:end_idx], dtype=np.float32)
    final_yaw = np.array(temp_yaw[start_idx:end_idx], dtype=np.float32)
    
    # Zero Origin
    start_x, start_y = final_pos[0]
    final_pos -= [start_x, start_y]

    traj_dir = os.path.join(dest_folder, f"trajectory_{traj_id}")
    ensure_dir(traj_dir)

    with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump({"position": final_pos, "yaw": final_yaw}, f)

    for i in range(len(final_pos)):
        src = images[start_idx + i]
        dst = os.path.join(traj_dir, f"{i}.jpg")
        shutil.copy2(src, dst)
        
    return True

def main():
    print(f"[*] Generating Final Hybrid-Calibrated Dataset (Scale={VELOCITY_SCALE}, WB={WHEELBASE})")
    if os.path.exists(OUTPUT_DATASET): shutil.rmtree(OUTPUT_DATASET)
    ensure_dir(OUTPUT_DATASET)

    config_path = os.path.join(OUTPUT_DATASET, "data_config.yaml")
    with open(config_path, "w") as f:
        f.write(f"metric_waypoint_spacing: {METRIC_WAYPOINT_SPACING}\n")
        f.write(f"fps: {1.0/DT}\n")

    recordings = [d for d in glob(os.path.join(SOURCE_DATASET, "capture-*")) if os.path.isdir(d)]
    
    count = 0
    for i, rec in enumerate(tqdm(recordings)):
        if generate_trajectory(rec, OUTPUT_DATASET, i):
            count += 1
            
    print(f"[*] Done. Processed {count} trajectories.")

if __name__ == "__main__":
    main()