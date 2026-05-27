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
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import cv2
from torchvision import transforms

# --- CONFIGURATION ---
# Point this to your Snowflake dataset root
DATASET_ROOT = r"C:\Users\chris\Embrained\embrained-app\data\1D\SnowflakePrimary"
JSON_FILE = "all_transitions.json"

# Import TinyVAE from your existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from modules.spatial_model import TinyVAE
except ImportError:
    print("Error: Could not import 'modules.spatial_model'. Run this from your app root.")
    sys.exit(1)

# Image Transformations (Same as your app)
IMG_H, IMG_W = 64, 64
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

def load_data_sequentially(root, json_name):
    """Loads transitions preserving chronological order."""
    path = os.path.join(root, json_name)
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        sys.exit(1)
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} transitions from {json_name}")
    return data

def load_vae(root):
    """Finds and loads the VAE encoder."""
    name = os.path.basename(root)
    # Look for the specific Snowflake encoder
    model_path = os.path.join(root, f"{name}_vae_encoder.pth")
    if not os.path.exists(model_path):
        # Fallback to generic name
        model_path = os.path.join(root, "vae_encoder.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: No VAE model found in {root}")
        sys.exit(1)

    print(f"Loading VAE from: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyVAE(latent_dim=32).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle dictionary mismatch (if saved as full model vs just encoder)
    try:
        if 'encoder.0.weight' in state_dict:
            model.load_state_dict(state_dict, strict=False)
        else:
            model.encoder.load_state_dict(state_dict)
    except:
        pass # Try proceeding if keys match partially
        
    model.eval()
    return model, device

def main():
    # 1. Setup
    transitions = load_data_sequentially(DATASET_ROOT, JSON_FILE)
    model, device = load_vae(DATASET_ROOT)
    
    # 2. Sequential Encoding
    latents = []
    print("Encoding images sequentially (this may take a moment)...")
    
    # We take every Nth frame to speed up visualization if dataset is huge, 
    # but for 14k transitions, we can do all or skip=2
    step = 5 
    subset = transitions[::step]
    
    with torch.no_grad():
        for i, t in enumerate(subset):
            img_path = os.path.join(DATASET_ROOT, t['image_path'])
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t_img = transform(img).unsqueeze(0).to(device)
            
            # Extract Mu (the latent mean)
            _, mu, _ = model(t_img)
            latents.append(mu.cpu().numpy().flatten())
            
            if i % 500 == 0:
                print(f"Processed {i}/{len(subset)} frames...")

    latents = np.array(latents)
    
    # 3. Dimensionality Reduction (PCA) to 2D
    print("Running PCA...")
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(latents)
    
    # 4. Generate the "Cool Graph"
    plt.figure(figsize=(10, 8), facecolor='#1e1e1e')
    ax = plt.axes()
    ax.set_facecolor('#1e1e1e')
    
    # Color by TIME (index) to show the winding trajectory
    # 'twilight' or 'hsv' are great for cyclical data
    scatter = ax.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=range(len(embedding)), 
        cmap='twilight', 
        s=15, 
        alpha=0.6,
        edgecolor='none'
    )
    
    # Styling
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time (Frame Index)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.title("Snowflake Topology Test: 1D Latent Manifold", color='white', fontsize=16)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)", color='gray')
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)", color='gray')
    plt.tick_params(colors='gray')
    plt.grid(color='gray', linestyle=':', alpha=0.3)
    
    # Output
    save_path = "snowflake_topology.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()