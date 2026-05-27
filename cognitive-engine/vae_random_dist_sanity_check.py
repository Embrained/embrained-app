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

import sys
print("Starting script...", flush=True)
import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.spatial_model import TinyVAE
from config import IMG_H, IMG_W

def load_data(data_root):
    # Find all markov_* directories
    markov_dirs = glob.glob(os.path.join(data_root, 'markov_*'))
    print(f"Found {len(markov_dirs)} markov_* directories.")
    
    image_paths = []
    for d in markov_dirs:
        images_dir = os.path.join(d, 'images')
        if os.path.exists(images_dir):
            imgs = glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png'))
            image_paths.extend(imgs)
            
    print(f"Total images found: {len(image_paths)}")
    return image_paths

def main():
    base_dir = r"c:\Users\chris\Embrained\software_suite"
    data_root = os.path.join(base_dir, "data")
    model_path = os.path.join(base_dir, "data", "tinyvae-vae_20260404_211254.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    latent_dim, model_size, input_spatial_dim, in_channels = TinyVAE.detect_size(state_dict)
    print(f"Detected model size: {model_size}, latent_dim: {latent_dim}, spatial_dim: {input_spatial_dim}, in_channels: {in_channels}")
    
    model = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load Data
    image_paths = load_data(data_root)
    if not image_paths:
        print("No images found. Exiting.")
        return
        
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((input_spatial_dim, input_spatial_dim)),
        transforms.ToTensor(),
    ])
    
    latents = []
    print("Encoding images into latents...")
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                batch_tensors.append(transform(img))
            
            batch_tensor = torch.stack(batch_tensors).to(device)
            _, mu, _ = model(batch_tensor)
            latents.append(mu.cpu().numpy())
            
    latents = np.concatenate(latents, axis=0) # shape: (N, latent_dim)
    print(f"Latents shape: {latents.shape}")
    
    N = latents.shape[0]
    
    # Pick 5 random images
    random_indices = random.sample(range(N), 5)
    letters = ['A', 'B', 'C', 'D', 'E']
    
    print("Generating grids for 5 random seed images...")
    for idx, letter in zip(random_indices, letters):
        seed_latent = latents[idx]
        seed_img_path = image_paths[idx]
        
        # Select 19 images at random for each seed
        available_indices = list(range(N))
        available_indices.remove(idx)
        random_19_indices = random.sample(available_indices, 19)
        
        # Calculate distances of these 19 to the seed
        random_19_latents = latents[random_19_indices]
        distances_19 = np.linalg.norm(random_19_latents - seed_latent, axis=-1)
        
        # Sort them by Euclidian distance to the seed
        sorted_args = np.argsort(distances_19)
        sorted_19_indices = [random_19_indices[i] for i in sorted_args]
        sorted_distances = distances_19[sorted_args]
        
        # Display in a 5x4 grid
        fig, axes = plt.subplots(n_rows:=4, n_cols:=5, figsize=(15, 12))
        fig.suptitle(f"Seed Image {letter} & 19 Random Images Sorted by Dist", fontsize=16)
        
        axes = axes.flatten()
        
        # Seed image
        seed_img = Image.open(seed_img_path)
        axes[0].imshow(seed_img)
        axes[0].set_title(f"Seed ({letter})\nDist: 0.0000", fontweight='bold')
        axes[0].axis('off')
        
        for spine in axes[0].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(5)
            spine.set_visible(True)
        axes[0].axis('on')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        for i, (neighbor_idx, dist) in enumerate(zip(sorted_19_indices, sorted_distances)):
            ax = axes[i+1]
            neighbor_img = Image.open(image_paths[neighbor_idx])
            ax.imshow(neighbor_img)
            ax.set_title(f"Dist: {dist:.4f}")
            ax.axis('off')
            
        plt.tight_layout()
        grid_filename = os.path.join(base_dir, f"vae_random_dist_grid_{letter}.png")
        plt.savefig(grid_filename)
        plt.close()
        
    print("Done! Saved grid visualizations to main directory.")

if __name__ == "__main__":
    main()
