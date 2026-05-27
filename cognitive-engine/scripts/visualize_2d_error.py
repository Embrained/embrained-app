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
# Point this to your Livingroom dataset
DATASET_ROOT = r"C:\Users\chris\Embrained\embrained-app\livingroom"
JSON_FILE = "all_transitions.json"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from modules.spatial_model import TinyVAE
except ImportError:
    print("Error: Could not import 'modules.spatial_model'.")
    sys.exit(1)

IMG_H, IMG_W = 64, 64
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

def load_vae(root):
    # Try specific 2D model name first, then generic
    name = os.path.basename(root)
    candidates = [
        os.path.join(root, f"{name}_vae_encoder.pth"),
        os.path.join(root, "livingroom_vae_encoder.pth"),
        os.path.join(root, "vae_encoder.pth")
    ]
    
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if not model_path:
        print(f"Error: No VAE model found in {root}")
        sys.exit(1)

    print(f"Loading VAE from: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyVAE(latent_dim=32).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    try:
        # Try full model load first
        model.load_state_dict(state_dict, strict=False)
    except:
        # Fallback to encoder only (common in your saves)
        try: model.encoder.load_state_dict(state_dict)
        except: pass
        
    model.eval()
    return model, device

def main():
    # 1. Load Data
    json_path = os.path.join(DATASET_ROOT, JSON_FILE)
    with open(json_path, 'r') as f:
        transitions = json.load(f)
    
    model, device = load_vae(DATASET_ROOT)
    
    latents = []
    errors = []
    
    # 2. Process a subset (e.g., every 5th frame) to keep it fast
    subset = transitions[::5]
    print(f"Processing {len(subset)} frames...")

    with torch.no_grad():
        for i, t in enumerate(subset):
            img_path = os.path.join(DATASET_ROOT, t['image_path'])
            if not os.path.exists(img_path): continue
            
            # Load and Preprocess
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            # Pass through VAE (Encode -> Decode)
            recon, mu, _ = model(input_tensor)
            
            # Calculate MSE (Mean Squared Error) per image
            loss = torch.mean((recon - input_tensor) ** 2).item()
            
            latents.append(mu.cpu().numpy().flatten())
            errors.append(loss)

    # 3. PCA Projection
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(latents)
    
    # 4. Plot
    plt.figure(figsize=(10, 8), facecolor='#1e1e1e')
    ax = plt.axes()
    ax.set_facecolor('#1e1e1e')
    
    # Color by RECONSTRUCTION ERROR (magma is good for "heat")
    scatter = ax.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=errors, 
        cmap='magma', 
        s=10, 
        alpha=0.7,
        edgecolor='none'
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Visual "Confusion" (MSE Loss)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.title("2D Blind Spot Map: Where does the robot struggle?", color='white')
    plt.axis('off')
    
    plt.savefig("livingroom_blindspots.png", dpi=150, bbox_inches='tight')
    print("Graph saved to livingroom_blindspots.png")
    plt.show()

if __name__ == "__main__":
    main()