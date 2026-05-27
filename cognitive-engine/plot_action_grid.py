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
import random
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
        print("Error: Could not load the CQL Network weights.")
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
            valid_samples.append(valid_frame)

    if len(valid_samples) > 3000:
        random.seed(42)
        valid_samples = random.sample(valid_samples, 3000)

    transform = transforms.Compose([
        transforms.Resize((input_spatial_dim, input_spatial_dim)),
        transforms.ToTensor(),
    ])

    actions = []
    batch_size = 64
    print("Encoding images to latents and inferring actions...")

    with torch.no_grad():
        for i in range(0, len(valid_samples), batch_size):
            batch_samples = valid_samples[i:i+batch_size]
            batch_tensors = []
            for filepath in batch_samples:
                t = transform(Image.open(filepath).convert('RGB'))
                batch_tensors.append(t)

            batch_tensor = torch.stack(batch_tensors).to(device)
            _, mu, _ = model(batch_tensor)

            # Predict Actions (Synthetic stationarity by tripling identical frame)
            mu_stacked = torch.cat([mu, mu, mu], dim=1)
            q_values = policy(mu_stacked)
            acts = q_values.argmax(dim=1).cpu().numpy()
            actions.extend(acts.tolist())

    actions = np.array(actions)
    
    # Target Actions
    target_actions = {
        1: "Forward",
        2: "Reverse",
        3: "Left",
        4: "Right"
    }
    
    # 4 rows, 5 columns grid
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle("Visualizing Policy Outputs: Random Frames for Each Action", fontsize=20, y=0.98)
    
    for row_idx, (act_id, act_name) in enumerate(target_actions.items()):
        # Find indices of all frames that predicted this action
        matching_indices = np.where(actions == act_id)[0]
        
        # Select up to 5 random samples
        num_samples = min(5, len(matching_indices))
        if num_samples > 0:
            selected_indices = random.sample(list(matching_indices), num_samples)
        else:
            selected_indices = []
            
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(selected_indices):
                frame_path = valid_samples[selected_indices[col_idx]]
                img = Image.open(frame_path)
                ax.imshow(img)
            else:
                # Empty placeholder if not enough samples
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.axis('off')
                
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Row labels on the first column
            if col_idx == 0:
                ax.set_ylabel(f"{act_name}", fontsize=14, rotation=0, labelpad=40, va='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(goals_dir, "action_perception_grid.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
