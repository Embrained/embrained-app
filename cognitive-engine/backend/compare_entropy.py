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
import numpy as np
from PIL import Image
from scipy.stats import entropy
import argparse
import random

# Configuration
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def load_images(dataset_name, sample_size=100):
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_name} not found at {dataset_path}")
        return []

    # Check for capture folders or root images
    # Using recursive glob as per VAEDataset fallback
    files = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
    files.extend(glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True))
    
    print(f"[{dataset_name}] Found {len(files)} images.")
    
    if len(files) > sample_size:
        files = random.sample(files, sample_size)
        print(f"[{dataset_name}] Sampled {len(files)} images.")
    
    images = []
    for f in files:
        try:
            img = Image.open(f).convert('RGB').resize((64, 64))
            images.append(np.array(img))
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return np.array(images)

def calculate_image_entropy(image):
    # Flatten image and calculate histogram
    flattened = image.flatten()
    hist, _ = np.histogram(flattened, bins=256, range=(0, 255))
    prob = hist / hist.sum()
    return entropy(prob, base=2)

def calculate_dataset_stats(dataset_name):
    print(f"\nAnalyzing {dataset_name}...")
    images = load_images(dataset_name)
    
    if len(images) == 0:
        return None

    # 1. Per-image Entropy (Richness)
    entropies = [calculate_image_entropy(img) for img in images]
    avg_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    
    # 2. Pairwise Diversity (L1/L2 distance between random pairs)
    # Normed by image size
    num_pairs = 1000
    l2_dists = []
    
    flat_images = images.reshape(len(images), -1) / 255.0
    
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(len(images), 2, replace=False)
        dist = np.linalg.norm(flat_images[idx1] - flat_images[idx2])
        l2_dists.append(dist)
        
    avg_dist = np.mean(l2_dists)
    
    print(f"Stats for {dataset_name}:")
    print(f"  Average Per-Image Entropy: {avg_entropy:.4f} +/- {std_entropy:.4f}")
    print(f"  Average Pairwise L2 Distance (Diversity): {avg_dist:.4f}")
    
    return {
        "avg_entropy": avg_entropy,
        "diversity": avg_dist
    }

if __name__ == "__main__":
    nook_stats = calculate_dataset_stats("nook")
    livingroom_stats = calculate_dataset_stats("livingroom")
    
    if nook_stats and livingroom_stats:
        print("\n=== COMPARISON ===")
        print(f"Entropy (Richness): Livingroom is {livingroom_stats['avg_entropy']/nook_stats['avg_entropy']:.2f}x that of Nook")
        print(f"Diversity: Livingroom is {livingroom_stats['diversity']/nook_stats['diversity']:.2f}x that of Nook")
