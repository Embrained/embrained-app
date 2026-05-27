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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
import umap

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
    model_path = os.path.join(base_dir, "data", "tinyvae-vae_20260311_163736.pth")
    
    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    state_dict = torch.load(model_path, map_location=device)
    latent_dim, model_size, input_spatial_dim = TinyVAE.detect_size(state_dict)
    print(f"Detected model size: {model_size}, latent_dim: {latent_dim}, spatial_dim: {input_spatial_dim}")
    
    model = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Load Data
    image_paths = load_data(data_root)
    if not image_paths:
        print("No images found. Exiting.")
        return
        
    # We will process in batches to avoid OOM
    batch_size = 256
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
    
    # Calculate average euclidean distance between all latents (approximate by sampling if too large)
    N = latents.shape[0]
    sample_size = min(N, 5000)
    indices = np.random.choice(N, sample_size, replace=False)
    sampled_latents = latents[indices]
    
    # Efficient pairwise distance computation
    dists = pdist(sampled_latents, metric='euclidean')
    avg_dist = np.mean(dists)
    print(f"Average latent distance: {avg_dist:.4f}")
    
    # 3. Dimensionality Reduction
    print("Running PCA...")
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    print("Running UMAP...")
    reducer = umap.UMAP()
    latents_umap = reducer.fit_transform(latents)
    
    print("Running t-SNE...")
    # t-SNE can be slow for large N, using a subset if necessary, but we'll try full datset or max 10k
    tsne_sample_size = min(N, 10000)
    tsne_indices = np.random.choice(N, tsne_sample_size, replace=False)
    tsne = TSNE(n_components=2, n_jobs=-1, random_state=42)
    latents_tsne_partial = tsne.fit_transform(latents[tsne_indices])
    # However we need the positions for the 5 random images in t-SNE, so we use full if possible, or just the subset.
    # Actually TSNE for 20-30k could take a while but is doable. Let's run it on the full set.
    latents_tsne = TSNE(n_components=2, n_jobs=-1, random_state=42).fit_transform(latents)
    
    # 4. Pick 5 random images
    random_indices = random.sample(range(N), 5)
    letters = ['A', 'B', 'C', 'D', 'E']
    
    def plot_embedding(embedding, title, filename):
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.5, c='gray')
        
        for idx, letter in zip(random_indices, letters):
            x, y = embedding[idx]
            plt.scatter(x, y, c='red', s=50, edgecolors='black', zorder=5)
            plt.annotate(letter, (x, y), xytext=(5, 5), textcoords='offset points', 
                         fontsize=12, fontweight='bold', color='black')
                         
        plt.title(f"{title}\nAvg Dist: {avg_dist:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, filename))
        plt.close()

    plot_embedding(latents_pca, "PCA", "vae_sanity_pca.png")
    plot_embedding(latents_umap, "UMAP", "vae_sanity_umap.png")
    plot_embedding(latents_tsne, "t-SNE", "vae_sanity_tsne.png")
    
    # 5. Nearest Neighbors for the 5 images
    print("Generating grids for 5 random images...")
    for idx, letter in zip(random_indices, letters):
        seed_latent = latents[idx]
        seed_img_path = image_paths[idx]
        
        # Calculate distances to all latents
        distances = np.linalg.norm(latents - seed_latent, axis=-1)
        # Get indices of 19 closest (excluding the seed itself, which is distance 0)
        # argsort returns indices sorted by distance
        closest_indices = np.argsort(distances)[1:20]
        
        # Plot 5x4 grid
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle(f"Seed Image {letter} and 19 Closest Neighbors in Latent Space", fontsize=16)
        
        axes = axes.flatten()
        
        # Seed image
        seed_img = Image.open(seed_img_path)
        axes[0].imshow(seed_img)
        axes[0].set_title(f"Seed {letter}\nDist: 0.00", fontweight='bold')
        axes[0].axis('off')
        
        # Add bold line around seed image
        for spine in axes[0].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(5)
            spine.set_visible(True)
        # Matplotlib imshow doesn't show spines easily without removing axis off, so we draw a rectangle
        axes[0].axis('on')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        for i, neighbor_idx in enumerate(closest_indices):
            ax = axes[i+1]
            neighbor_path = image_paths[neighbor_idx]
            neighbor_img = Image.open(neighbor_path)
            ax.imshow(neighbor_img)
            dist = distances[neighbor_idx]
            ax.set_title(f"Dist: {dist:.4f}")
            ax.axis('off')
            
        plt.tight_layout()
        grid_filename = os.path.join(base_dir, f"vae_sanity_grid_{letter}.png")
        plt.savefig(grid_filename)
        plt.close()
        
    print("Done! Saved visualizations and grids to main directory.")

if __name__ == "__main__":
    main()
