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
import sys
import glob
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import itertools

# Add root path so we can import modules
import pathlib
root_path = str(pathlib.Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from modules.spatial_model import TinyVAE
from backend.services.datasets import DatasetService

def analyze_markov_data(dataset_path, vae_path, output_path):
    print(f"Loading VAE from {vae_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VAE
    state_dict = torch.load(vae_path, map_location=device)
    latent_dim, model_size, input_spatial_dim, _ = TinyVAE.detect_size(state_dict)
    print(f"Detected VAE: size={model_size}, latent={latent_dim}, spatial={input_spatial_dim}")
    
    vae = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim).to(device)
    vae.load_state_dict(state_dict)
    vae.eval()
    
    transform = T.Compose([
        T.Resize((input_spatial_dim, input_spatial_dim)),
        T.ToTensor()
    ])
    
    print(f"Loading transitions from {dataset_path}...")
    ds_service = DatasetService(data_root=os.path.dirname(dataset_path))
    transitions = ds_service.load_transitions(dataset_path)
    print(f"Loaded {len(transitions)} transitions.")
    
    def get_latent(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                # For TinyVAE, forward returns (recon, mu, logvar)
                recon, mu, logvar = vae(t)
                return mu.squeeze(0).cpu() # use mean as the deterministic point
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None

    # Group by action
    dz_by_action = {}
    
    print("Computing latents and differences...")
    for i in range(len(transitions) - 1):
        t1 = transitions[i]
        t2 = transitions[i+1]
        
        # Ensure they are sequential frames
        if str(t1.get('format')) != 'markov':
            continue
            
        action = t1.get('macro_action')
        
        # Load images and get latents
        z1 = get_latent(t1['image_path'])
        z2 = get_latent(t2['image_path'])
        
        if z1 is not None and z2 is not None:
            dz = z2 - z1
            if action not in dz_by_action:
                dz_by_action[action] = []
            dz_by_action[action].append(dz)

    # Analyze Action 1 (Forward)
    if 1 not in dz_by_action or len(dz_by_action[1]) == 0:
        print("No Action 1 (Forward) transitions found.")
        return
        
    dz_forward = torch.stack(dz_by_action[1])
    n = len(dz_forward)
    print(f"Found {n} transitions for Action 1 (Forward).")
    
    print("Calculating cosine similarities...")
    # Normalize vectors
    dz_norm = dz_forward / torch.norm(dz_forward, dim=1, keepdim=True)
    
    # Compute dot product (cosine similarity since normalized)
    cos_sim_matrix = torch.mm(dz_norm, dz_norm.t())
    
    # Extract upper triangle (excluding diagonal) to get unique pairs
    triu_indices = torch.triu_indices(n, n, offset=1)
    similarities = cos_sim_matrix[triu_indices[0], triu_indices[1]].numpy()
    
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    print(f"Mean Cosine Similarity for Forward Actions: {mean_sim:.4f} (std: {std_sim:.4f})")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    plt.title('Cosine Similarity between $\Delta z$ vectors for Action 1 (Forward)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Markov Data Consistency")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the markov dataset directory")
    parser.add_argument("--vae", type=str, required=True, help="Path to the trained VAE model .pth file")
    parser.add_argument("--output", type=str, default="action1_consistency.png", help="Output plot image path")
    args = parser.parse_args()
    
    analyze_markov_data(args.dataset, args.vae, args.output)
