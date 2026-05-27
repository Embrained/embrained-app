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
import glob
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Set up relative imports for vision system
import sys
sys.path.append(os.path.dirname(__file__))
from modules.vision import VisionSystem

def load_images_from_dir(directory, limit=None):
    files = sorted(glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.png")))
    if limit is not None and len(files) > limit:
        files = random.sample(files, limit)
    
    images = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            images.append(img)
    return images

def main():
    print("Loading Vision System...")
    # Load the latest VAE model
    models_dir = os.path.join(os.path.dirname(__file__), "data")
    vae_models = sorted(glob.glob(os.path.join(models_dir, "tinyvae-vae_*.pth")))
    if not vae_models:
        print("No VAE model found.")
        return
        
    latest_vae = vae_models[-1]
    print(f"Using VAE: {latest_vae}")
    
    vision = VisionSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    vision.load_model(latest_vae)
    
    print("Loading Images...")
    # Load background images
    bg_dir = os.path.join(models_dir, "random") 
    if not os.path.exists(bg_dir):
         # Find any dataset goals dir
         goal_dirs = glob.glob(os.path.join(models_dir, "..", "data", "*_goals"))
         if goal_dirs:
             bg_dir = goal_dirs[0]
         else:
             print("No background images found.")
             return
         print(f"Using {bg_dir} for background images")
         
    bg_imgs = load_images_from_dir(bg_dir, limit=1000)
    print(f"Loaded {len(bg_imgs)} background images.")
    
    # Load patrol images
    patrol_dir = os.path.join(models_dir, "patrol")
    patrol_a_files = sorted(glob.glob(os.path.join(patrol_dir, "patrol_a*.jpg")))
    patrol_b_files = sorted(glob.glob(os.path.join(patrol_dir, "patrol_b*.jpg")))
    
    imgs_a = [cv2.imread(f) for f in patrol_a_files if cv2.imread(f) is not None]
    imgs_b = [cv2.imread(f) for f in patrol_b_files if cv2.imread(f) is not None]
    
    print(f"Loaded {len(imgs_a)} A target images and {len(imgs_b)} B target images.")
    
    print("Encoding Images...")
    def encode_set(imgs):
        latents = []
        for img in imgs:
            _, z = vision.process_frame(img)
            if z is not None:
                latents.append(z)
        return np.array(latents)
        
    z_bg = encode_set(bg_imgs)
    z_a = encode_set(imgs_a)
    z_b = encode_set(imgs_b)
    
    if len(z_bg) == 0 or len(z_a) == 0 or len(z_b) == 0:
        print("Failed to encode all image sets.")
        return
        
    # Combine for dimensionality reduction
    X_all = np.vstack([z_bg, z_a, z_b])
    
    # Compute Distances in Latent Space (Original 32D)
    # Average distance between all background latents
    # (Sampling 1000 pairs to avoid huge memory spike)
    idx1 = np.random.randint(0, len(z_bg), 10000)
    idx2 = np.random.randint(0, len(z_bg), 10000)
    bg_dists = np.linalg.norm(z_bg[idx1] - z_bg[idx2], axis=1)
    avg_bg_dist = np.mean(bg_dists)
    
    # Intra-cluster distances
    dist_a = np.mean([np.linalg.norm(p1 - p2) for p1 in z_a for p2 in z_a if not np.array_equal(p1, p2)])
    dist_b = np.mean([np.linalg.norm(p1 - p2) for p1 in z_b for p2 in z_b if not np.array_equal(p1, p2)])
    
    # Inter-cluster distance
    dist_ab = np.mean([np.linalg.norm(p1 - p2) for p1 in z_a for p2 in z_b])
    
    print(f"\n--- Latent Space Distances (32D) ---")
    print(f"Avg pairwise dist (Background): {avg_bg_dist:.4f}")
    print(f"Avg intra-cluster dist (A):     {dist_a:.4f} ({(dist_a/avg_bg_dist)*100:.1f}%)")
    print(f"Avg intra-cluster dist (B):     {dist_b:.4f} ({(dist_b/avg_bg_dist)*100:.1f}%)")
    print(f"Avg inter-cluster dist (A vs B):{dist_ab:.4f} ({(dist_ab/avg_bg_dist)*100:.1f}%)")
    
    print("\nRunning Dimensionality Reduction...")
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_all)
    
    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_all)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Patrol Goals Latent Space Analysis', fontsize=20, y=1.05)
    
    reductions = [
        ("PCA", X_pca),
        ("t-SNE", X_tsne),
        ("UMAP", X_umap)
    ]
    
    n_bg = len(z_bg)
    n_a = len(z_a)
    n_b = len(z_b)
    
    for ax, (title, X_red) in zip(axes, reductions):
        # Create an individual figure for each plot
        fig_ind, ax_ind = plt.subplots(figsize=(8, 8))
        
        for curr_ax in [ax, ax_ind]:
            # Background
            curr_ax.scatter(X_red[:n_bg, 0], X_red[:n_bg, 1], c='grey', alpha=0.3, s=10, label='Background')
            
            # A goals
            curr_ax.scatter(X_red[n_bg:n_bg+n_a, 0], X_red[n_bg:n_bg+n_a, 1], c='orange', alpha=0.9, s=50, marker='o', edgecolors='white', linewidth=0.5, label='Patrol A')
            
            # B goals
            curr_ax.scatter(X_red[n_bg+n_a:, 0], X_red[n_bg+n_a:, 1], c='purple', alpha=0.9, s=50, marker='o', edgecolors='white', linewidth=0.5, label='Patrol B')
            
            # Centroids
            centroid_a = np.mean(X_red[n_bg:n_bg+n_a], axis=0)
            centroid_b = np.mean(X_red[n_bg+n_a:], axis=0)
            curr_ax.scatter(centroid_a[0], centroid_a[1], c='orange', marker='*', s=300, edgecolors='black')
            curr_ax.scatter(centroid_b[0], centroid_b[1], c='purple', marker='*', s=300, edgecolors='black')
            
            curr_ax.set_title(f"{title} - Latent Space", fontsize=16)
            curr_ax.grid(True, alpha=0.2)
            if title == "PCA":
                curr_ax.legend(fontsize=12)

        # Save individual figure
        ind_out_path = os.path.join(os.path.dirname(__file__), f"patrol_{title.lower()}_analysis.png")
        fig_ind.savefig(ind_out_path, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)
        print(f"Saved {title} plot to {ind_out_path}")

    # Add text annotations with distances
    stats_text = (
        f"32D Latent Metrics relative to Avg. Background Dist ({avg_bg_dist:.2f}):\n"
        f"A Spread (intra): {(dist_a/avg_bg_dist)*100:.1f}%\n"
        f"B Spread (intra): {(dist_b/avg_bg_dist)*100:.1f}%\n"
        f"A-B Distance (inter): {(dist_ab/avg_bg_dist)*100:.1f}%"
    )
    plt.figtext(0.5, 0.01, stats_text, wrap=True, horizontalalignment='center', fontsize=14,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10, 'edgecolor': 'gray'})

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'patrol_clustering_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()
