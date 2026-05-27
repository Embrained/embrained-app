import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.models.latentslam import LatentSLAM

def generate_reachability_plots():
    data_dir = r"C:\Users\chris\Embrained\software_suite\data"
    output_dir = r"C:\Users\chris\.gemini\antigravity\brain\f38af09f-2bd0-489f-9f34-71894172bea0"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data...")
    
    # Load Telemetry
    telemetry_path = os.path.join(data_dir, "master_telemetry.csv")
    df = pd.read_csv(telemetry_path)
    df['ts'] = df['ts'].astype(str)
    telemetry_dict = df.set_index('ts')[['cx', 'cy']].to_dict('index')
    
    import glob
    
    # Dynamically find the most recent global latents cache
    latent_files = glob.glob(os.path.join(data_dir, "*_global_latents.pt"))
    if not latent_files:
        print("No global latents found.")
        return
    latent_files.sort(key=os.path.getmtime, reverse=True)
    global_latents_path = latent_files[0]
    
    latent_data = torch.load(global_latents_path, map_location='cpu', weights_only=True)
    ts_map = latent_data.get("ts_map", {})
    
    # Align valid nodes
    valid_ts = []
    latents = []
    coords = []
    
    for ts, l in ts_map.items():
        if ts in telemetry_dict:
            valid_ts.append(ts)
            latents.append(l.squeeze().numpy())
            coords.append([telemetry_dict[ts]['cx'], telemetry_dict[ts]['cy']])
            
    latents = np.array(latents)
    coords = np.array(coords)
    num_nodes = len(valid_ts)
    print(f"Aligned {num_nodes} latents with telemetry.")
    
    if num_nodes == 0:
        print("No matches. Exiting.")
        return

    # Find Latest Forward Model
    fwd_files = glob.glob(os.path.join(data_dir, "topological_forward_latentslam_*.pth"))
    if not fwd_files:
        print("No Forward Model found.")
        return
    fwd_files.sort(key=os.path.getmtime, reverse=True)
    model_path = fwd_files[0]
    
    print(f"Loading Base Latents: {os.path.basename(global_latents_path)}")
    print(f"Loading Forward Model: {os.path.basename(model_path)}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = LatentSLAM(latent_dim=128, model_size='large', image_size=64, num_actions=3, hidden_dim=1024).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Predict Next States
    print("Evaluating Action-Transitions across Manifold...")
    latents_t = torch.tensor(latents, dtype=torch.float32).to(device)
    batch_size = 1000
    all_preds = []
    with torch.no_grad():
        for i in range(0, num_nodes, batch_size):
            batch = latents_t[i:i+batch_size]
            preds = model.predict_next_state(batch) # [B, 3, 128]
            all_preds.append(preds.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) # [N, 3, 128]
    
    # Build Graph Using GPU Torch CDist instead of KDTree
    print("Building Nearest-Neighbor Transition Graph (GPU)...")
    
    edges_from = []
    edges_to = []
    weights = []
    
    latents_gpu = torch.tensor(latents, dtype=torch.float32, device=device)
    all_preds_gpu = torch.tensor(all_preds, dtype=torch.float32, device=device)
    
    for a in range(3):
        preds_a = all_preds_gpu[:, a, :]
        
        # Batch cdist
        chunk_size = 500
        indices = []
        distances = []
        for i in range(0, num_nodes, chunk_size):
            chunk = preds_a[i:i+chunk_size]
            dist_matrix = torch.cdist(chunk, latents_gpu)
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
    
    # Pre-compute PCA for plotting once
    print("Computing PCA...")
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    # Generate Plots
    num_seeds = 5
    seed_indices = random.sample(range(num_nodes), num_seeds)
    artifact_paths = []
    
    for idx, seed_i in enumerate(seed_indices):
        print(f"Processing Seed {idx+1}/{num_seeds}...")
        
        # Run Dijkstra
        dist_matrix = dijkstra(csgraph=adj_matrix, directed=True, indices=seed_i, return_predecessors=False)
        valid_mask = ~np.isinf(dist_matrix)
        
        # Normalize finite distances for colormap
        max_dist = np.percentile(dist_matrix[valid_mask], 95) # clip extreme outliers
        plot_dists = np.clip(dist_matrix, 0, max_dist)
        plot_dists[~valid_mask] = max_dist + (max_dist*0.1) # make unreachable distinct
        
        # Plot 1: Manifold
        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')
        sc = plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=plot_dists, cmap='magma_r', s=5, alpha=0.7)
        plt.scatter(latents_pca[seed_i, 0], latents_pca[seed_i, 1], c='lime', s=200, marker='*', edgecolor='white', label='Start Location')
        plt.colorbar(sc, label='Reachability Effort (Hops + Predict Confidence)')
        plt.title(f"Manifold Reachability Graph (Seed {idx+1})")
        plt.legend()
        manifold_path = os.path.join(output_dir, f"reachability_seed_{idx+1}_manifold.png")
        plt.savefig(manifold_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifact_paths.append(manifold_path)
        
        # Plot 2: Overhead
        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=plot_dists, cmap='magma_r', s=10, alpha=0.7)
        plt.scatter(coords[seed_i, 0], coords[seed_i, 1], c='lime', s=300, marker='*', edgecolor='white', label='Start Location')
        plt.colorbar(sc, label='Reachability Effort (Hops + Predict Confidence)')
        plt.title(f"Physical Map Reachability (Seed {idx+1})")
        plt.legend()
        overhead_path = os.path.join(output_dir, f"reachability_seed_{idx+1}_overhead.png")
        plt.savefig(overhead_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifact_paths.append(overhead_path)
        
    print("Done! Artifact paths generated.")

if __name__ == "__main__":
    generate_reachability_plots()
