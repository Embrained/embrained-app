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
import numpy as np
import pickle
import torch
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize

# ================= CONFIGURATION =================
RAW_DATA_DIR = os.path.join("data", "livingroom")
SAMPLE_RATE = 10     # Match every 10th frame (Speed optimization)
DT = 0.1

# Initial Guesses (Your current "Guesswork")
INITIAL_V_SCALE = 1.0   # Multiplier for your current gearbox
INITIAL_WHEELBASE = 0.14 # Meters

# Fixed "Gearbox" Ratios (We will scale these globally)
# These represent the RELATIVE difference between PWM levels
GEAR_1 = 0.05
GEAR_2 = 0.09
GEAR_3 = 0.12
# =================================================

def parse_motor_cmd(cmd_str):
    if not isinstance(cmd_str, str): return 0, 0
    match = re.search(r'l:(-?\d+);r:(-?\d+);', cmd_str)
    if match: return int(match.group(1)), int(match.group(2))
    return 0, 0

def get_base_velocity(pwm):
    """ Returns the UN-SCALED velocity from the gearbox """
    if abs(pwm) < 40: return 0.0
    mag = abs(pwm)
    if mag <= 100: speed = GEAR_1
    elif mag <= 200: speed = GEAR_2
    else: speed = GEAR_3
    
    # Direction
    vel = speed if pwm > 0 else -speed
    return vel

def load_raw_logs():
    """ Load ALL raw commands into memory for fast re-integration """
    print("[*] Loading raw logs...")
    trajectory_commands = []
    
    dirs = natsorted([d for d in glob(os.path.join(RAW_DATA_DIR, "capture-*")) if os.path.isdir(d)])
    
    for d in dirs:
        log_path = os.path.join(d, "log.csv")
        cmds = []
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                # Normalize headers
                reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
                for row in reader:
                    l, r = parse_motor_cmd(row.get('motor_cmd', ''))
                    # Store tuple: (v_left_base, v_right_base)
                    # Note: We handle LEFT inversion here for simplicity
                    vl = -get_base_velocity(l) # Left Inverted
                    vr = get_base_velocity(r)
                    cmds.append((vl, vr))
        trajectory_commands.append(cmds)
    return trajectory_commands, dirs

def find_visual_constraints(log_dirs):
    """ Re-runs the visual match finding to get constraints """
    print("[*] Finding Visual Constraints (this takes a minute)...")
    
    # Load Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval()
    if torch.cuda.is_available(): model = model.cuda()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    embeddings = []
    metadata = [] # (traj_idx, frame_idx)
    
    # Extract
    for t_idx, folder in enumerate(tqdm(log_dirs)):
        images = natsorted(glob(os.path.join(folder, "*.jpg")))
        for i in range(0, len(images), SAMPLE_RATE):
            try:
                img = Image.open(images[i]).convert("RGB")
                tensor = preprocess(img).unsqueeze(0)
                if torch.cuda.is_available(): tensor = tensor.cuda()
                with torch.no_grad():
                    emb = model(tensor).cpu().numpy().flatten()
                embeddings.append(emb)
                metadata.append((t_idx, i))
            except: pass
            
    # Match
    emb_array = np.array(embeddings)
    sim_matrix = cosine_similarity(emb_array)
    np.fill_diagonal(sim_matrix, 0)
    
    matches = []
    rows, cols = np.where(sim_matrix > 0.94) # Strict threshold
    
    for r, c in zip(rows, cols):
        if r >= c: continue
        tA, fA = metadata[r]
        tB, fB = metadata[c]
        
        # Only cross-trajectory matches or distant self-matches
        if tA != tB or abs(fA - fB) > 50:
            matches.append(((tA, fA), (tB, fB)))
            
    print(f"[*] Found {len(matches)} constraints.")
    return matches

def simulate_trajectory(commands, v_scale, wheelbase):
    """ Fast integration of a single trajectory """
    x, y, theta = 0.0, 0.0, 0.0
    path = []
    
    # Pre-calculate constants
    # v_robot = (vl + vr) / 2 * v_scale
    # w_robot = (vr - vl) * v_scale / wheelbase
    
    for vl, vr in commands:
        path.append([x, y])
        
        v = (vl + vr) / 2.0 * v_scale
        w = (vr - vl) * v_scale / wheelbase
        
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        theta += w * DT
        
    return np.array(path)

def cost_function(params, trajectory_commands, matches):
    """ 
    The Objective: Minimize distance between matched frames 
    Updated Limit: Reduced from 0.1 to 0.01 to test for physical minima.
    """
    v_scale, wheelbase = params
    
    # NEW LIMITS: Allow v_scale to go down to 0.01
    # We still need >0 to prevent division by zero or singularity
    if v_scale < 0.01 or v_scale > 5.0 or wheelbase < 0.05 or wheelbase > 0.5:
        return 1e9
        
    total_error = 0
    
    # 1. Integrate ALL trajectories with current params
    trajs = []
    for cmds in trajectory_commands:
        trajs.append(simulate_trajectory(cmds, v_scale, wheelbase))
        
    # 2. Sum errors
    for (tA, fA), (tB, fB) in matches:
        try:
            posA = trajs[tA][fA]
            posB = trajs[tB][fB]
            dist = np.linalg.norm(posA - posB)
            total_error += dist
        except IndexError:
            pass 
            
    return total_error / len(matches)

def main():
    # 1. Prep Data
    trajectory_commands, log_dirs = load_raw_logs()
    matches = find_visual_constraints(log_dirs)
    
    if not matches:
        print("[!] No matches found. Cannot calibrate.")
        return

    print(f"[*] Starting Optimization with {len(matches)} visual anchors...")
    print(f"    Initial Guess: V_Scale={INITIAL_V_SCALE}, Wheelbase={INITIAL_WHEELBASE}")
    
    # 2. Optimize
    # We want to find [v_scale, wheelbase]
    initial_guess = [INITIAL_V_SCALE, INITIAL_WHEELBASE]
    
    res = minimize(
        cost_function, 
        initial_guess, 
        args=(trajectory_commands, matches),
        method='Nelder-Mead',
        tol=1e-2
    )
    
    # 3. Report
    print("\n" + "="*40)
    print("CALIBRATION COMPLETE")
    print("="*40)
    print(f"Optimization Success: {res.success}")
    print(f"Final Cost (Avg Error): {res.fun:.4f} meters")
    
    opt_v_scale, opt_wheelbase = res.x
    
    print("\nOPTIMIZED PARAMETERS:")
    print(f"   VELOCITY_MULTIPLIER : {opt_v_scale:.4f}")
    print(f"   EFFECTIVE_WHEELBASE : {opt_wheelbase:.4f}")
    
    print("\n" + "-"*40)
    print("ACTION: Update process_data_vint.py with these new values.")
    print(f"1. Multiply your GEAR speeds (0.05, 0.09, 0.12) by {opt_v_scale:.4f}")
    print(f"   -> New Gear 1: {0.05 * opt_v_scale:.4f}")
    print(f"   -> New Gear 2: {0.09 * opt_v_scale:.4f}")
    print(f"   -> New Gear 3: {0.12 * opt_v_scale:.4f}")
    print(f"2. Set WHEELBASE = {opt_wheelbase:.4f}")
    print("-"*40)

if __name__ == "__main__":
    main()