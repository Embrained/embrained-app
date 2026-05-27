import os
import sys
import json
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.models.latentslam import LatentSLAM

def generate_image_reachability():
    data_dir = r"C:\Users\chris\Embrained\software_suite\data"
    output_dir = r"C:\Users\chris\.gemini\antigravity\brain\f38af09f-2bd0-489f-9f34-71894172bea0"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data...")
    
    # Load transition map for image paths
    trans_path = os.path.join(data_dir, "all_transitions.json")
    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    ts_to_img = {}
    for node in all_data:
        ip = node.get('image_path', '')
        if ip:
            ts = os.path.basename(ip).replace('frame_', '').replace('.jpg', '')
            ts_to_img[ts] = os.path.join(data_dir, ip)
            
    # Load Latents
    latent_files = glob.glob(os.path.join(data_dir, "*_global_latents.pt"))
    latent_files.sort(key=os.path.getmtime, reverse=True)
    global_latents_path = latent_files[0]
    
    latent_data = torch.load(global_latents_path, map_location='cpu', weights_only=True)
    ts_map = latent_data.get("ts_map", {})
    
    valid_ts = []
    latents = []
    
    for ts, l in ts_map.items():
        if ts in ts_to_img and os.path.exists(ts_to_img[ts]):
            valid_ts.append(ts)
            latents.append(l.squeeze().numpy())
            
    latents = np.array(latents)
    num_nodes = len(valid_ts)
    print(f"Aligned {num_nodes} latents with images.")
    
    if num_nodes == 0: return

    # Find Latest Forward Model
    fwd_files = glob.glob(os.path.join(data_dir, "topological_forward_latentslam_*.pth"))
    fwd_files.sort(key=os.path.getmtime, reverse=True)
    model_path = fwd_files[0]
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = LatentSLAM(latent_dim=128, model_size='large', image_size=64, num_actions=3, hidden_dim=1024).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Evaluating Action-Transitions...")
    latents_t = torch.tensor(latents, dtype=torch.float32).to(device)
    batch_size = 1000
    all_preds = []
    with torch.no_grad():
        for i in range(0, num_nodes, batch_size):
            batch = latents_t[i:i+batch_size]
            preds = model.predict_next_state(batch)
            all_preds.append(preds.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) # [N, 3, 128]
    
    print("Building Nearest-Neighbor Transition Graph (GPU)...")
    edges_from = []
    edges_to = []
    weights = []
    
    all_preds_gpu = torch.tensor(all_preds, dtype=torch.float32, device=device)
    
    for a in range(3):
        preds_a = all_preds_gpu[:, a, :]
        chunk_size = 500
        indices = []
        distances = []
        for i in range(0, num_nodes, chunk_size):
            chunk = preds_a[i:i+chunk_size]
            dist_matrix = torch.cdist(chunk, latents_t)
            min_dist, min_idx = torch.min(dist_matrix, dim=1)
            distances.append(min_dist.cpu().numpy())
            indices.append(min_idx.cpu().numpy())
            
        distances = np.concatenate(distances)
        indices = np.concatenate(indices)
        
        edges_from.extend(np.arange(num_nodes))
        edges_to.extend(indices)
        costs = 1.0 + (5.0 * distances)
        weights.extend(costs)
        
    adj_matrix = csr_matrix((weights, (edges_from, edges_to)), shape=(num_nodes, num_nodes))
    
    num_seeds = 3
    seed_indices = random.sample(range(num_nodes), num_seeds)
    
    for idx, seed_i in enumerate(seed_indices):
        print(f"Processing Seed {idx+1}/{num_seeds}...")
        
        # Run Dijkstra
        dist_matrix = dijkstra(csgraph=adj_matrix, directed=True, indices=seed_i, return_predecessors=False)
        
        # Sample 1000 nodes randomly (excluding self)
        sample_pool = list(range(num_nodes))
        sample_pool.remove(seed_i)
        sub_sample = random.sample(sample_pool, min(1000, len(sample_pool)))
        
        # Extract distances for these 1000 nodes
        sample_dists = [(si, dist_matrix[si]) for si in sub_sample if not np.isinf(dist_matrix[si])]
        
        # Sort by distance
        sample_dists.sort(key=lambda x: x[1])
        
        # Extract Top 10 Easy
        easy_nodes = sample_dists[:10]
        
        # Extract Bottom 10 Hard (or completely disconnected ones)
        # We also want to find physically disconnected nodes if possible
        disconnected = [si for si in sub_sample if np.isinf(dist_matrix[si])]
        hard_nodes = sample_dists[-10:] if len(disconnected) < 10 else [(x, float('inf')) for x in disconnected[:10]]
        
        # If we have less than 10 easy nodes for some reason
        if len(easy_nodes) < 10:
            print(f"  Warning: Only {len(easy_nodes)} easy nodes found.")
            
        # Helper to load image
        def load_img(index):
            ts = valid_ts[index]
            path = ts_to_img[ts]
            # Replace absolute path if we have to
            if not os.path.exists(path):
                # Try relative
                path = os.path.join(data_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path))
            return Image.open(path).convert("RGB")
            
        seed_img = load_img(seed_i)
        
        fig = plt.figure(figsize=(15, 8))
        fig.patch.set_facecolor('#1a1a2e')
        plt.axis('off')
        
        # Add Seed Image
        ax_seed = fig.add_subplot(3, 1, 1)
        ax_seed.imshow(seed_img)
        ax_seed.axis('off')
        ax_seed.set_title("SEED LOCATION (Current State)", color='white', pad=10, fontsize=14, fontweight='bold')
        
        # Add 10 Easy
        for j, (ei, d) in enumerate(easy_nodes):
            ax = fig.add_subplot(3, 10, 10 + j + 1)
            ax.imshow(load_img(ei))
            ax.axis('off')
            color = 'lime'
            ax.set_title(f"Score: {d:.1f}", color=color, fontsize=10)
            if j == 0:
                ax.text(-0.1, 0.5, "EASY", color='lime', fontsize=14, fontweight='bold', transform=ax.transAxes, ha='right', va='center')
                
        # Add 10 Hard
        for j, (hi, d) in enumerate(hard_nodes):
            ax = fig.add_subplot(3, 10, 20 + j + 1)
            ax.imshow(load_img(hi))
            ax.axis('off')
            color = 'red'
            dist_str = "INF" if np.isinf(d) else f"{d:.1f}"
            ax.set_title(f"Score: {dist_str}", color=color, fontsize=10)
            if j == 0:
                ax.text(-0.1, 0.5, "HARD", color='red', fontsize=14, fontweight='bold', transform=ax.transAxes, ha='right', va='center')
                
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"visual_reachability_seed_{idx+1}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
    print("Done generating visual charts!")

if __name__ == "__main__":
    generate_image_reachability()
