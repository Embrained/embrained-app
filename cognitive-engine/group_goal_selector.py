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
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
    
    # We use the requested latest VAE
    model_name = "tinyvae-vae_20260405_114442.pth"
    model_path = os.path.join(data_root, model_name)
    telemetry_path = os.path.join(base_dir, "master_telemetry.csv")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find VAE at {model_path}")
        return
        
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
            valid_samples.append({
                'frame': valid_frame,
                'ts': ts_str
            })
            
    print(f"Found {len(valid_samples)} valid samples.")
    if len(valid_samples) == 0:
        print("No valid samples. Exiting.")
        return
        
    # Downsample if needed to prevent memory issues
    if len(valid_samples) > 2000:
        import random
        random.seed(42)
        valid_samples = random.sample(valid_samples, 2000)

    transform = transforms.Compose([
        transforms.Resize((input_spatial_dim, input_spatial_dim)),
        transforms.ToTensor(),
    ])
    
    # Pre-encode all valid images
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

    # UI State
    selected_idx = [0] # List used for reference mutation
    current_image_artist = [None]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Plot manifold scatter
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c='#ff8c00', edgecolors='#000000', alpha=0.7, picker=True, pickradius=5)
    
    # Highlight point
    highlight, = ax.plot([], [], 'o', color='#00ff00', markersize=10, fillstyle='none', markeredgewidth=2)
    
    ax.set_title("Manifold Group Goal Selector", color='black', fontsize=16)
    ax.grid(color='#cccccc', linestyle='--', alpha=0.5)
    ax.tick_params(colors='black')
        
    # Inset Display Placeholder
    imagebox = OffsetImage(np.zeros((64, 64, 3)), zoom=2.0)
    ab = AnnotationBbox(imagebox, (0, 0), xybox=(20, -20), xycoords='data', boxcoords="offset points", pad=0.1, frameon=True, bboxprops=dict(edgecolor='orange', lw=2))
    ax.add_artist(ab)
    ab.set_visible(False)
    
    def update_inset(idx):
        selected_idx[0] = idx
        highlight.set_data([latents_2d[idx, 0]], [latents_2d[idx, 1]])
        
        # Load full resolution image for UI
        img_path = valid_samples[idx]['frame']
        img_ui = Image.open(img_path).convert('RGB')
        img_ui.thumbnail((128, 128))
        
        imagebox.set_data(np.array(img_ui))
        ab.xybox = (10, 10)
        ab.xy = (latents_2d[idx, 0], latents_2d[idx, 1])
        ab.set_visible(True)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax: return
        
        # Brute force NN in 2D space so it feels responsive to the mouse cursor click position
        click_coord = np.array([event.xdata, event.ydata])
        distances_2d = np.linalg.norm(latents_2d - click_coord, axis=1)
        nearest = np.argmin(distances_2d)
        
        update_inset(nearest)
        
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Extraction Logic Callback
    def extract_group(event):
        idx = selected_idx[0]
        if idx is None:
            print("Please click the manifold to select a target first.")
            return
            
        print(f"\n--- EXTRACTING GROUP GOAL ---")
        base_latent = latents[idx]
        
        # 1. High-dim nearest neighbors
        dists_high_dim = np.linalg.norm(latents - base_latent, axis=1)
        
        # 2. Get top 20 (19 nearest + the 1 itself)
        top_20_indices = np.argsort(dists_high_dim)[:20]
        
        # 3. Clean goal dir
        if not os.path.exists(goals_dir):
            os.makedirs(goals_dir)
        else:
            for f in os.listdir(goals_dir):
                file_path = os.path.join(goals_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    
        # 4. Save frames and compute stats
        copied_files = []
        group_latents = []
        
        for i, neighbor_idx in enumerate(top_20_indices):
            src = valid_samples[neighbor_idx]['frame']
            ext = os.path.splitext(src)[1]
            dest = os.path.join(goals_dir, f"group_goal_{i:02d}{ext}")
            shutil.copy2(src, dest)
            copied_files.append(dest)
            group_latents.append(latents[neighbor_idx])
            
        group_latents = np.array(group_latents) # [20, D]
        centroid = np.mean(group_latents, axis=0)
        
        # Compute average euclidean distance to centroid FOR THE IN-GROUP
        in_group_distances = np.linalg.norm(group_latents - centroid, axis=1)
        avg_distance = float(np.mean(in_group_distances))
        
        # Save Stats
        stats_path = os.path.join(goals_dir, "group_stats.json")
        with open(stats_path, 'w') as f:
            json.dump({
                "type": "group_goal",
                "vae_model": model_name,
                "centroid": centroid.tolist(),
                "average_in_group_distance": avg_distance,
                "num_members": 20
            }, f, indent=4)
            
        print(f"Goal generation successful!")
        print(f"Extracted {len(copied_files)} closest images to {goals_dir}")
        print(f"Average In-Group Distance: {avg_distance:.4f}")
        print(f"Threshold for Training Reward: {(avg_distance * 1.1):.4f}")
        print(f"Centroid dimensionality: {centroid.shape[0]}")
        print(f"-----------------------------\n")
        
        # Visual feedback on plot
        highlight.set_data(latents_2d[top_20_indices, 0], latents_2d[top_20_indices, 1])
        highlight.set_color('#00ffff')
        fig.canvas.draw_idle()

    # Add button
    ax_btn = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn = Button(ax_btn, 'Extract Group (20)', color='#ff8c00', hovercolor='#ffd700')
    btn.on_clicked(extract_group)

    print("\nInitialization complete. Opening interactive interface...")
    print("Click anywhere on the manifold to snap to the nearest visual transition.")
    print("Click 'Extract Group' to export the region for CQL Training.")
    
    plt.show()

if __name__ == "__main__":
    main()
