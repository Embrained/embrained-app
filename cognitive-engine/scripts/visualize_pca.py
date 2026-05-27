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
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import cv2
import torch

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules.vision import VisionSystem
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.vision import VisionSystem

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_all_latents(root_dir, num_samples=None):
    logging.info(f"Searching for latents in {root_dir}")
    
    # search for all latents.npy files in capture-* directories
    pattern = os.path.join(root_dir, "capture-*", "latents.npy")
    latent_files = glob.glob(pattern)
    
    if not latent_files:
        logging.warning("No latents.npy files found.")
        return None
        
    all_latents = []
    
    for f in latent_files:
        try:
            data = np.load(f)
            if data.ndim == 2 and data.shape[1] == 576:
                all_latents.append(data)
            else:
                logging.warning(f"Skipping {f}: Unexpected shape {data.shape}")
        except Exception as e:
            logging.error(f"Error loading {f}: {e}")
            
    if not all_latents:
        return None
        
    # Concatenate all
    combined = np.concatenate(all_latents, axis=0)
    logging.info(f"Loaded total {combined.shape[0]} latents.")
    
    if num_samples is not None and num_samples < combined.shape[0]:
        logging.info(f"Randomly sampling {num_samples} latents...")
        indices = np.random.choice(combined.shape[0], num_samples, replace=False)
        combined = combined[indices]
        
    return combined

def get_goal_latent(vision_system, image_path):
    if not os.path.exists(image_path):
        logging.error(f"Goal image not found: {image_path}")
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Failed to read goal image: {image_path}")
        return None
        
    _, latent = vision_system.process_frame(img)
    return latent

def plot_pca(pca_model, all_latents, goal_latent=None, goal_name="Goal"):
    # Transform all latents
    latents_2d = pca_model.transform(all_latents)
    
    plt.figure(figsize=(10, 8))
    
    # Plot background points
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.5, s=2, label='Captured Frames', c='blue')
    
    if goal_latent is not None:
        # Transform goal
        goal_2d = pca_model.transform(goal_latent.reshape(1, -1))
        
        # Plot goal
        plt.scatter(goal_2d[:, 0], goal_2d[:, 1], s=200, c='red', marker='*', label=goal_name)
        
        # Calculate distances (Euclidean in original space)
        # Using the original space for distance because PCA distorts/loses info
        # But user said "distance from the goal frame (i.e. latent)" and "use this frame of reference"
        # Often PCA is just for viz.
        
        # Let's compute min distance in original space to any captured frame just for info
        dists = np.linalg.norm(all_latents - goal_latent, axis=1)
        min_dist = np.min(dists)
        avg_dist = np.mean(dists)
        
        plt.title(f"PCA Visualization (Goal Min Dist: {min_dist:.4f}, Avg: {avg_dist:.4f})")
    else:
        plt.title("PCA Visualization of Latent Space")
        
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%} var)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Visualize latents with PCA.")
    parser.add_argument("--root_dir", type=str, default=r"C:\Users\chris\ArtificialBrain\Explorer", help="Root directory containing capture folders")
    parser.add_argument("--goal_image", type=str, help="Path to a goal image to plot")
    parser.add_argument("-n", "--num_samples", type=int, default=None, help="Number of random latents to visualize.")
    args = parser.parse_args()
    
    # 1. Load Data
    all_latents = load_all_latents(args.root_dir, num_samples=args.num_samples)
    if all_latents is None:
        logging.error("Could not load any latents. Run generate_capture_latents.py first.")
        return

    # 2. Fit PCA
    logging.info("Fitting PCA...")
    pca = PCA(n_components=2)
    pca.fit(all_latents)
    
    # 3. Process Goal if provided
    goal_latent = None
    if args.goal_image:
        # N.B. Need VisionSystem to encode goal image
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
           vision = VisionSystem(device=device)
           goal_latent = get_goal_latent(vision, args.goal_image)
        except Exception as e:
            logging.error(f"Failed to initialize vision system or process goal: {e}")

    # 4. Plot
    plot_pca(pca, all_latents, goal_latent, goal_name=os.path.basename(args.goal_image) if args.goal_image else "Goal")

if __name__ == "__main__":
    main()
