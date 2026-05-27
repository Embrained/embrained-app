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
import argparse
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_H, IMG_W, ACTION_DIM, HIDDEN_DIM
from modules.spatial_model import TinyVAE, CQLNetwork

def angular_distance(yaw1, yaw2):
    """Returns the signed angular distance from yaw1 to yaw2 in degrees.
       Positive means CCW (Left), Negative means CW (Right)."""
    diff = (yaw2 - yaw1) % 360
    if diff > 180:
        diff -= 360
    return diff

def verify_cql_policy(vae_path, cql_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    telemetry_path = "master_telemetry.csv"
    if not os.path.exists(telemetry_path):
        print("master_telemetry.csv not found!")
        return
        
    df = pd.read_csv(telemetry_path)
    df = df[(df['ir'] > 0) & (df['dist_px'] > 0)].reset_index(drop=True)
    
    # 1. Load VAE
    print(f"Loading VAE from {vae_path}...")
    vae_state = torch.load(vae_path, map_location=device, weights_only=True)
    latent_dim, model_size, input_spatial_dim, _ = TinyVAE.detect_size(vae_state)
    vae = TinyVAE(model_size=model_size, latent_dim=latent_dim, input_spatial_dim=input_spatial_dim).to(device)
    vae.load_state_dict(vae_state)
    vae.eval()
    
    # 2. Load CQL
    print(f"Loading CQL Policy from {cql_path}...")
    cql_checkpoint = torch.load(cql_path, map_location=device, weights_only=True)
    state_dict = cql_checkpoint.get('model_state_dict', cql_checkpoint)
    
    # Dynamically detect policy input dimension based on current CQLNetwork init
    # Policy expects 3 historical current frames (latent) + 1 goal frame (latent)
    policy_input_dim = latent_dim * 4
    
    # Dynamically detect policy size
    cql = None
    for size in ['tiny', 'small', 'medium', 'large']:
        try:
            test_cql = CQLNetwork(input_dim=policy_input_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, model_size=size).to(device)
            test_cql.load_state_dict(state_dict)
            cql = test_cql
            print(f"Successfully fit CQL state dictionary to model size: {size.upper()}")
            break
        except RuntimeError:
            pass
            
    if cql is None:
        raise ValueError("Could not fit weights to any standard model size.")
        
    cql.eval()
    
    base_dir = r"c:\Users\chris\Embrained\software_suite\data"
    transform = transforms.Compose([
        transforms.Resize((input_spatial_dim, input_spatial_dim)),
        transforms.ToTensor(),
    ])
    
    # Extract latents into memory array
    print("Pre-computing latent space mapped to telemetry indices...")
    latents = []
    valid_indices = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            found = False
            for d in os.listdir(base_dir):
                if 'markov' in d:
                    ts_str = str(int(float(row['ts'])))
                    path = os.path.join(base_dir, d, 'images', f"webcam_frame_{ts_str}.jpg")
                    if os.path.exists(path):
                        img = Image.open(path).convert('RGB')
                        t_img = transform(img).unsqueeze(0).to(device)
                        _, mu, _ = vae(t_img)
                        latents.append(mu.squeeze(0).cpu())
                        valid_indices.append(idx)
                        found = True
                        break
    
    latents_stack = torch.stack(latents) # [N, latent_dim]
    valid_df = df.loc[valid_indices].reset_index(drop=True)
    
    N = len(valid_df)
    print(f"Loaded {N} strictly valid physical frames.")
    
    num_samples = 3000
    results = {'dist': [], 'correct': [], 'q_left': [], 'q_right': [], 'pred_action': []}
    
    print(f"Simulating {num_samples} random spatial RL goals...")
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Target goal index anywhere
            i_goal = np.random.randint(0, N)
            
            # Start position needs at least 2 frames of history contiguous (index - 2)
            # We strictly enforce i_start >= 2 and randomly bound it
            i_start = np.random.randint(2, N)
            
            # Ensure they aren't trivially the same explicit position
            if abs(i_start - i_goal) < 2:
                continue
                
            # Physics distance
            yaw_start = valid_df.iloc[i_start]['yaw_deg']
            yaw_goal = valid_df.iloc[i_goal]['yaw_deg']
            diff = angular_distance(yaw_start, yaw_goal)
            
            # Only test rotation macro-actions
            if abs(diff) < 20: 
                continue # Ambiguous forward/stop
                
            correct_action = 3 if diff > 0 else 4 # 3=Left (CCW), 4=Right (CW)
            
            # Input Stack: [t-2, t-1, t, goal] (Optical Flow History Stack)
            z_curr_2 = latents_stack[i_start - 2]
            z_curr_1 = latents_stack[i_start - 1]
            z_curr = latents_stack[i_start] # [32]
            z_goal = latents_stack[i_goal] # [32]
            
            state_tensor = torch.cat([z_curr_2, z_curr_1, z_curr, z_goal]).unsqueeze(0).to(device) # [1, 128]
            
            q_vals = cql(state_tensor).squeeze(0) # [5]
            
            pred_action = torch.argmax(q_vals).item()
            is_correct = (pred_action == correct_action)
            
            results['dist'].append(diff)
            results['correct'].append(1 if is_correct else 0)
            results['q_left'].append(q_vals[3].item())
            results['q_right'].append(q_vals[4].item())
            results['pred_action'].append(pred_action)

    # Plotting Correlation Validation
    results_df = pd.DataFrame(results)
    
    print("\n--- Action Prediction Bias Histogram ---")
    print(results_df['pred_action'].value_counts(normalize=True))
    print("--------------------------------------\n")
    
    # Bin by angle to compute explicit policy accuracy curve
    bins = np.linspace(-180, 180, 20)
    results_df['bin'] = pd.cut(results_df['dist'], bins=bins)
    binned_acc = results_df.groupby('bin')['correct'].mean().fillna(0)
    bin_centers = [b.mid for b in binned_acc.index]
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, binned_acc.values * 100, marker='o', linestyle='-', color='b', lw=2)
    plt.axvline(0, color='r', linestyle='--', alpha=0.5, label='Optimal Threshold')
    plt.title(f"CQL Spatial Rotation Benchmark\nTotal Synthetic Transitions: {len(results_df)}", fontsize=14)
    plt.xlabel("Physical Angular Distance to Goal (Degrees)\nNegative = Target is CW (Right) | Positive = Target is CCW (Left)", fontsize=12)
    plt.ylabel("Policy Generalization Accuracy (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    
    # Highlight Left / Right
    plt.axvspan(-180, -20, color='red', alpha=0.05, label='Right Action (4) Required')
    plt.axvspan(20, 180, color='green', alpha=0.05, label='Left Action (3) Required')
    plt.legend()
    
    out_file = "cql_spatial_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"\nSpatial Sanity benchmark complete! Saved overlay to {out_file}")
    
    total_acc = results_df['correct'].mean() * 100
    print(f"Overall Rotation Generalization Accuracy: {total_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_pth", type=str, required=True)
    parser.add_argument("--cql_pth", type=str, required=True)
    args = parser.parse_args()
    
    verify_cql_policy(args.vae_pth, args.cql_pth)
