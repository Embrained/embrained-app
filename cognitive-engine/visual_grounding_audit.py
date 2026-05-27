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
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
try:
    from natsort import natsorted
except ImportError:
    # Fallback if natsort is not installed
    import re
    def natsorted(l): 
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        print("Processing...")
        return iterable

from torchvision import transforms, models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# CONFIGURATION
DATASET_DIR = os.path.join("data", "vint_formatted_livingroom")
SAMPLE_RATE = 10  # Only check every 10th frame (Process ~15k images, not 150k)
SIMILARITY_THRESHOLD = 0.92  # Very strict! Only "identical" images.
# Lower threshold slightly if using ImageNet weights directly without fine-tuning, usually 0.90-0.95 is good for duplicates.

def load_encoder():
    """ Load a pre-trained lightweight CNN to act as our 'Eye' """
    print("[*] Loading Visual Encoder (EfficientNet-B0)...")
    # Using weights=DEFAULT which is usually IMAGENET1K_V1 or V2
    model = models.efficientnet_b0(weights='DEFAULT')
    model.classifier = torch.nn.Identity() # Remove classification layer, keep embeddings
    model.eval()
    
    device = "cpu"
    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    
    print(f"[*] Model loaded on {device}")
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess, device

def get_embeddings(dataset_dir, model, preprocess, device):
    """ Extract visual features from the dataset """
    traj_folders = natsorted(glob(os.path.join(dataset_dir, "trajectory_*")))
    
    all_embeddings = []
    all_coords = []
    metadata = [] # Stores (traj_index, frame_index)

    print(f"[*] Extracting embeddings from {len(traj_folders)} trajectories...")
    
    with torch.no_grad():
        for t_idx, folder in enumerate(tqdm(traj_folders)):
            # Load Positions
            pkl_path = os.path.join(folder, "traj_data.pkl")
            if not os.path.exists(pkl_path):
                continue
                
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                pos = data["position"] # (N, 2)
            
            # Load Images
            img_paths = natsorted(glob(os.path.join(folder, "*.jpg")))
            
            # Subsample
            # Ensure we don't go out of bounds if pos and imgs have diff lengths
            limit = min(len(pos), len(img_paths))
            
            for i in range(0, limit, SAMPLE_RATE):
                img_path = img_paths[i]
                
                # Load & Preprocess
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = preprocess(img).unsqueeze(0)
                    if device == "cuda":
                        tensor = tensor.cuda()
                    
                    # Forward Pass
                    emb = model(tensor).cpu().numpy().flatten()
                    
                    all_embeddings.append(emb)
                    all_coords.append(pos[i])
                    metadata.append((t_idx, i))
                    
                except Exception as e:
                    # print(f"Error loading {img_path}: {e}")
                    pass

    return np.array(all_embeddings), np.array(all_coords), metadata

def visualize_drift(embeddings, coords, metadata):
    if len(embeddings) == 0:
        print("[!] No embeddings extracted.")
        return

    print("[*] Computing Pairwise Similarity (this may take a moment)...")
    
    # Compute Similarity Matrix (N x N)
    sim_matrix = cosine_similarity(embeddings)
    
    # Zero out diagonal and neighbors (don't match with self or immediate neighbors)
    # We want to ignore frames that are temporally close in the same trajectory (e.g. i and i+1)
    # Since we subsample by 10, adjacent indices in matrix are 10 frames apart.
    # We should ignore diagonals and maybe +/- 1 or 2 indices if in same traj.
    
    np.fill_diagonal(sim_matrix, 0)
    
    plt.figure(figsize=(12, 10))
    plt.title(f"Visual Grounding Audit\nGreen: Odometry Path | Red: Visual Loop Closures (Sim > {SIMILARITY_THRESHOLD})")
    
    # 1. Plot the Odometry (The "Map")
    plt.scatter(coords[:, 0], coords[:, 1], c='green', s=1, alpha=0.3, label="Odometry Path")
    
    # 2. Find and Plot Loop Closures
    num_matches = 0
    errors = []
    
    # Iterate through upper triangle of matrix
    # Use torch or np to find indices > threshold to check
    rows, cols = np.where(sim_matrix > SIMILARITY_THRESHOLD)
    
    # Use a set to track drawn pairs to avoid double drawing if upper triangle not strictly enforced by where?
    # np.where returns all matches. sim_matrix is symmetric.
    # We only assume r < c to do upper triangle.
    
    for r, c in zip(rows, cols):
        if r >= c: continue # Avoid duplicates and lower triangle
        
        # Get metadata
        traj_A, frame_A = metadata[r][0], metadata[r][1]
        traj_B, frame_B = metadata[c][0], metadata[c][1]
        
        # Filter: Ignore matches within the same trajectory if they are close temporally
        # Frame A and Frame B are raw frame indices.
        if traj_A == traj_B:
            if abs(frame_A - frame_B) < 100: # 100 frames = 10 seconds. Ignore immediate local similarity.
                continue
            
        # Get Coordinates
        pos_A = coords[r]
        pos_B = coords[c]
        
        # Calculate "Drift Error" (Distance between points that SHOULD be identical)
        dist = np.linalg.norm(pos_A - pos_B)
        errors.append(dist)
        
        # Draw "Spring" (Red Line)
        plt.plot([pos_A[0], pos_B[0]], [pos_A[1], pos_B[1]], c='red', alpha=0.1, linewidth=0.5)
        num_matches += 1

    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("visual_grounding_audit.png")
    
    print(f"[*] Analysis Complete.")
    print(f"    Total Visual Matches Found: {num_matches}")
    if errors:
        print(f"    Mean Odometry Error: {np.mean(errors):.2f} meters")
        print(f"    Max Odometry Error:  {np.max(errors):.2f} meters")
        print(f"[*] Saved plot to visual_grounding_audit.png")
    else:
        print("[!] No strong visual matches found. Try lowering threshold or checking image quality.")

if __name__ == "__main__":
    model, preprocess, device = load_encoder()
    emb, pos, meta = get_embeddings(DATASET_DIR, model, preprocess, device)
    visualize_drift(emb, pos, meta)
