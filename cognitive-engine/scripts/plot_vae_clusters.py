import argparse
import os
import sys
import json
import random
import torch
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR
from modules.vision import VisionSystem

def load_images_and_latents(vision_sys, max_samples=1000):
    transitions_path = os.path.join(DATA_DIR, "all_transitions.json")
    if not os.path.exists(transitions_path):
        raise FileNotFoundError(f"Could not find {transitions_path}")
        
    with open(transitions_path, 'r') as f:
        data = json.load(f)
        
    # Get unique image paths to avoid duplicates
    img_paths = list(set([t['image_path'] for t in data]))
    
    if len(img_paths) > max_samples:
        img_paths = random.sample(img_paths, max_samples)
        
    latents = []
    valid_paths = []
    images = []
    
    is_discrete = vision_sys.encoder is not None and 'Discrete' in type(vision_sys.encoder).__name__
    
    for p in img_paths:
        full_path = os.path.join(DATA_DIR, p)
        if not os.path.exists(full_path):
            continue
            
        img = cv2.imread(full_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if is_discrete:
            # For VQ-VAE: extract the continuous quantized embedding (z_q)
            # instead of one-hot, so PCA and distances are meaningful
            import torch
            tensor = vision_sys.transform(img_rgb).unsqueeze(0).to(vision_sys.device)
            with torch.no_grad():
                x_enc = vision_sys.encoder.encoder(tensor)
                z_e = vision_sys.encoder.fc_e(x_enc)
                vq = vision_sys.encoder.vq
                # Run VQ forward to get quantized embedding
                z_q, _, _, _ = vq(z_e)
                latent = z_q.cpu().numpy().flatten()
        else:
            # Continuous VAE: use process_frame as before
            _, latent = vision_sys.process_frame(img)
        
        if latent is not None:
            latents.append(latent)
            valid_paths.append(full_path)
            images.append(img_rgb)
            
    return np.array(latents), valid_paths, images

def main():
    parser = argparse.ArgumentParser(description="Plot VAE Clusters")
    parser.add_argument("--vae", type=str, required=True, help="Filename of the VAE (e.g. vqvae_512c_32d_20260426_173941.pth)")
    parser.add_argument("--samples", type=int, default=1500, help="Number of random samples to use")
    args = parser.parse_args()
    
    vae_path = os.path.join(DATA_DIR, args.vae)
    if not os.path.exists(vae_path):
        print(f"Error: VAE not found at {vae_path}")
        return
        
    print(f"Loading VAE: {args.vae}")
    vision = VisionSystem()
    if not vision.load_model(vae_path):
        print("Failed to load VAE.")
        return
        
    print(f"Extracting latents for {args.samples} random images...")
    latents, paths, images = load_images_and_latents(vision, max_samples=args.samples)
    
    if len(latents) == 0:
        print("No valid latents extracted.")
        return
        
    print(f"Running PCA on {len(latents)} latents...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    print("Selecting 5 random seeds...")
    num_seeds = 5
    num_neighbors = 5
    seed_indices = random.sample(range(len(latents)), num_seeds)
    
    colors = ['#FF3366', '#33CCFF', '#99FF33', '#FF9933', '#CC33FF']
    
    print("Generating plot...")
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], figure=fig)
    
    # Left: Manifold Plot
    ax_manifold = fig.add_subplot(gs[0, 0])
    ax_manifold.scatter(latents_2d[:, 0], latents_2d[:, 1], c='gray', alpha=0.3, s=20)
    
    # Right: Image Grid (5x6)
    gs_images = gridspec.GridSpecFromSubplotSpec(num_seeds, num_neighbors + 1, subplot_spec=gs[0, 1], wspace=0.1, hspace=0.3)
    
    for i, seed_idx in enumerate(seed_indices):
        seed_latent = latents[seed_idx]
        seed_2d = latents_2d[seed_idx]
        color = colors[i]
        
        # Highlight seed on manifold
        ax_manifold.scatter(seed_2d[0], seed_2d[1], c=color, s=300, edgecolors='black', marker='*', zorder=5)
        
        # Find distances to all other latents
        distances = np.linalg.norm(latents - seed_latent, axis=1)
        
        # Get nearest neighbors (excluding the seed itself)
        sorted_indices = np.argsort(distances)
        neighbor_indices = []
        for idx in sorted_indices:
            if idx != seed_idx:
                neighbor_indices.append(idx)
            if len(neighbor_indices) == num_neighbors:
                break
                
        # Plot seed image (Column 0)
        ax_img = fig.add_subplot(gs_images[i, 0])
        ax_img.imshow(images[seed_idx])
        ax_img.set_title(f"Seed {i+1}", color=color, fontweight='bold', fontsize=16)
        
        # Highlight border with color
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
        ax_img.axis('on')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Plot neighbors (Columns 1-5)
        for j, n_idx in enumerate(neighbor_indices):
            ax_img = fig.add_subplot(gs_images[i, j + 1])
            ax_img.imshow(images[n_idx])
            dist = distances[n_idx]
            ax_img.set_title(f"Dist: {dist:.4f}", fontsize=14)
            
            # Highlight border
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
            ax_img.axis('on')
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            # Plot neighbor on manifold as small dot
            n_2d = latents_2d[n_idx]
            ax_manifold.scatter(n_2d[0], n_2d[1], c=color, s=80, edgecolors='black', alpha=0.8, zorder=4)
            
    # Count unique latent vectors (for discrete: how many unique codebook tokens used)
    unique_latents = len(set([tuple(l) for l in latents.tolist()]))
    ax_manifold.set_title(f"Latent Manifold Clustering Analysis ({unique_latents} unique tokens / {len(latents)} samples)", fontsize=18)
    ax_manifold.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, f"vae_cluster_analysis_{args.vae.replace('.pth', '')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved cluster analysis to {out_path}")

if __name__ == "__main__":
    main()
