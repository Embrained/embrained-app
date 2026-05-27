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
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.spatial_model import TinyVAE

def main():
    base_dir = r"c:\Users\chris\Embrained\software_suite"
    data_root = os.path.join(base_dir, "data")
    goals_dir = os.path.join(data_root, "goals")
    
    # Load Stats
    stats_path = os.path.join(goals_dir, "group_stats.json")
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} does not exist. Please extract a group goal first using group_goal_selector.py.")
        return
        
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        
    centroid = np.array(stats['centroid'])
    avg_dist = stats['average_in_group_distance']
    threshold = avg_dist * 1.1
    model_name = stats.get('vae_model', "tinyvae-vae_20260405_114442.pth")
    
    print(f"Loaded Target Region: Centroid dim={centroid.shape[0]}, Threshold={threshold:.4f}")
    
    model_path = os.path.join(data_root, model_name)
    telemetry_path = os.path.join(base_dir, "master_telemetry.csv")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load VAE
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    latent_dim, model_size, input_spatial_dim, in_channels = TinyVAE.detect_size(state_dict)
    model = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load Physical Data
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
    batch_size = 64
    print("Encoding images to latent vectors...")
    
    with torch.no_grad():
        for i in range(0, len(valid_samples), batch_size):
            batch_samples = valid_samples[i:i+batch_size]
            batch_tensors = []
            for s in batch_samples:
                t = transform(Image.open(s['frame']).convert('RGB'))
                batch_tensors.append(t)
            
            batch_tensor = torch.stack(batch_tensors).to(device)
            _, mu, _ = model(batch_tensor)
            latents.append(mu.cpu().numpy())
            
    latents = np.concatenate(latents, axis=0) # [N, latent_dim]
    
    print("Computing PCA Projection...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    # Calculate Distances
    distances = np.linalg.norm(latents - centroid, axis=1)
    
    # Split into inside strict, inside loose, and outside
    inside_strict_idx = distances <= threshold
    inside_loose_idx = (distances > threshold) & (distances <= 2.0)
    outside_idx = distances > 2.0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Outside
    ax.scatter(latents_2d[outside_idx, 0], latents_2d[outside_idx, 1], 
               c='#bbbbbb', edgecolors='none', alpha=0.5, s=30, label='Outside Region')
               
    # Plot Inside Loose
    ax.scatter(latents_2d[inside_loose_idx, 0], latents_2d[inside_loose_idx, 1], 
               c='#ff8c00', edgecolors='#000000', alpha=0.7, s=40, label='Inside Loose (<=2.0)')
    
    # Plot Inside Strict
    ax.scatter(latents_2d[inside_strict_idx, 0], latents_2d[inside_strict_idx, 1], 
               c='#00ff00', edgecolors='#000000', alpha=0.9, s=50, label='Inside Strict (Target)')
    
    ax.set_title("CQL State Space: Sparse Reward Target Region", color='black', fontsize=16)
    ax.grid(color='#eeeeee', linestyle='--', alpha=0.9)
    ax.tick_params(colors='black')
        
    legend = ax.legend(loc='upper right')
    for text in legend.get_texts():
        text.set_color('black')
        
    # Add text annotation
    percentage_loose = (np.sum(inside_loose_idx) / len(valid_samples)) * 100
    percentage_strict = (np.sum(inside_strict_idx) / len(valid_samples)) * 100
    
    info_text = (f"Target Threshold: {threshold:.2f}\n"
                 f"Total Nodes: {len(valid_samples)}\n"
                 f"Nodes <= 2.0 Region: {np.sum(inside_loose_idx)} ({percentage_loose:.1f}%)\n"
                 f"Nodes in Strict Target: {np.sum(inside_strict_idx)} ({percentage_strict:.1f}%)")
                 
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, color='black',
            bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.5', alpha=0.9))

    output_file = os.path.join(goals_dir, "manifold_target_region.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
