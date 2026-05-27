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
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.spatial_model import TinyVAE, CQLNetwork
from config import ACTION_DIM, HIDDEN_DIM

def main():
    base_dir = r"c:\Users\chris\Embrained\software_suite"
    data_root = os.path.join(base_dir, "data")
    goals_dir = os.path.join(data_root, "goals")

    policy_filename = "tinyvae-vae_20260405_114442-group-goal-cql_20260406_204244.pth"
    policy_path = os.path.join(data_root, policy_filename)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Policy and Embedded Metadata
    checkpoint = torch.load(policy_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    centroid = checkpoint.get('group_centroid', None)
    if centroid is not None:
        centroid = np.array(centroid)
    threshold = checkpoint.get('group_avg_dist', 0.0) * 1.1

    print(f"Loaded Embedded Target Region natively from checkpoint! Centroid threshold bounds: {threshold:.4f}")

    # Load CQL Policy
    inp_dim = 96 # 3 frames stacked * 32 dim
    
    possible_sizes = ['tiny', 'small', 'medium', 'large', 'enormous', 'tectonic']
    loaded_policy = None
    for size in possible_sizes:
        try:
            test_policy = CQLNetwork(input_dim=inp_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, use_ln=True, model_size=size).to(device)
            test_policy.load_state_dict(state_dict, strict=True)
            loaded_policy = test_policy
            print(f"Loaded policy with size: {size}")
            break
        except Exception:
            pass
            
    if loaded_policy is None:
        print("Error: Could not dynamically load the CQL Network weights using any standard size.")
        return
        
    policy = loaded_policy
    policy.eval()

    # Load VAE (Fallback to explicit file)
    vae_model_name = "tinyvae-vae_20260405_114442.pth"
    vae_path = os.path.join(data_root, vae_model_name)
    vae_dict = torch.load(vae_path, map_location=device, weights_only=False)
    latent_dim, v_model_size, input_spatial_dim, in_channels = TinyVAE.detect_size(vae_dict)
    model = TinyVAE(latent_dim=latent_dim, model_size=v_model_size, input_spatial_dim=input_spatial_dim, in_channels=in_channels).to(device)
    model.load_state_dict(vae_dict)
    model.eval()

    # Load Physical Data
    telemetry_path = os.path.join(base_dir, "master_telemetry.csv")
    print("Loading telemetry...")
    df = pd.read_csv(telemetry_path)
    valid_samples = []

    for idx, row in df.iterrows():
        img_dir = row.get('img_dir', None)
        ts = row.get('ts', None)
        if pd.isna(ts) or not img_dir:
            continue

        try: ts_str = str(int(float(ts)))
        except: continue

        frame_jpg = os.path.join(img_dir, f"frame_{ts_str}.jpg")
        frame_png = os.path.join(img_dir, f"frame_{ts_str}.png")

        valid_frame = None
        if os.path.exists(frame_jpg): valid_frame = frame_jpg
        elif os.path.exists(frame_png): valid_frame = frame_png

        if valid_frame:
            valid_samples.append({'frame': valid_frame})

    if len(valid_samples) > 2000:
        import random
        random.seed(42)
        valid_samples = random.sample(valid_samples, 2000)

    transform = transforms.Compose([
        transforms.Resize((input_spatial_dim, input_spatial_dim)),
        transforms.ToTensor(),
    ])

    latents = []
    actions = []
    batch_size = 64
    print("Encoding images to latents and inferring actions...")

    with torch.no_grad():
        for i in range(0, len(valid_samples), batch_size):
            batch_samples = valid_samples[i:i+batch_size]
            batch_tensors = []
            for s in batch_samples:
                t = transform(Image.open(s['frame']).convert('RGB'))
                batch_tensors.append(t)

            batch_tensor = torch.stack(batch_tensors).to(device)
            _, mu, _ = model(batch_tensor)
            
            # Predict Actions (Synthetic stationarity by tripling identical frame)
            mu_np = mu.cpu().numpy()
            latents.append(mu_np)
            
            # Stack [N, 96]
            mu_stacked = torch.cat([mu, mu, mu], dim=1) 
            q_values = policy(mu_stacked)
            acts = q_values.argmax(dim=1).cpu().numpy()
            actions.append(acts)

    latents = np.concatenate(latents, axis=0) # [N, latent_dim]
    actions = np.concatenate(actions, axis=0) # [N]

    # Print Tally
    unique, counts = np.unique(actions, return_counts=True)
    tally = dict(zip(unique, counts))
    print(f"\nAction Tally for {len(actions)} points: {tally}\n")

    print("Computing PCA Projection...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    if centroid is not None:
        centroid_2d = pca.transform(centroid.reshape(1, -1))[0]
    else:
        centroid_2d = None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Action Mapping
    # 0=STOP (Red), 1=FWD (Blue), 2=REV (Black), 3=LEFT (Cyan), 4=RIGHT (Magenta)
    action_colors = {
        0: ('#ff4c4c', 'STOP'),
        1: ('#4c9eff', 'FWD'),
        2: ('#333333', 'REV'),
        3: ('#4cffd6', 'LEFT'),
        4: ('#ff4cee', 'RIGHT')
    }

    for action_id, (color, label) in action_colors.items():
        mask = (actions == action_id)
        if mask.any():
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                       c=color, edgecolors='#000000', alpha=0.7, s=40, label=label)

    # Plot Centroid Star
    if centroid_2d is not None:
        ax.scatter(centroid_2d[0], centroid_2d[1],
                   c='#00ff00', edgecolors='#000000', marker='*', s=400, label='Target Centroid', zorder=5)

    ax.set_title("Group Goal Navigation Policy Action Mapping", color='black', fontsize=16)
    ax.grid(color='#eeeeee', linestyle='--', alpha=0.9)
    ax.tick_params(colors='black')

    legend = ax.legend(loc='upper right')
    for text in legend.get_texts():
        text.set_color('black')

    info_text = (f"Model: {policy_filename}\n"
                 f"Actions Mapped: {len(actions)}")

    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, color='black',
            bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.5', alpha=0.9))

    output_file = os.path.join(goals_dir, "group_goal_action_manifold.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
