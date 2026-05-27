# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training Pipeline for Hello World: Fixed-Goal Offline Reinforcement Learning (CQL)
# specifically optimized for Discrete VQ-VAE latent topologies.

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_DIR
from backend.training.train_fixed_goal import TARGET_IMAGES, normalize_path
from backend.train_cql import train as run_cql_train
from modules.spatial_model import DiscreteVQVAE

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Discrete Fixed-Goal CQL Offline Reinforcement Learning on {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    target_normalized = [normalize_path(p) for p in TARGET_IMAGES]
        
    # Dynamically find the latest VQ-VAE model
    print("Searching for latest Discrete VQ-VAE checkpoint in data directory...")
    vae_candidates = [f for f in glob.glob(os.path.join(DATA_ROOT, 'vqvae_*.pth')) if 'discrete_cql_model' not in f]
    if not vae_candidates:
        print("Error: No Discrete VQ-VAE found matching pattern 'vqvae_*.pth'!")
        return
        
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    VAE_PATH = vae_candidates[0]
    vae_basename = os.path.basename(VAE_PATH).replace('.pth', '')
    print(f"-> Selected latest VQ-VAE: {os.path.basename(VAE_PATH)}")

    # Parse hyperparameters from filename: vqvae_{num_embeddings}c_{latent_dim}d_{timestamp}.pth
    parts = vae_basename.split('_')
    try:
        num_embeddings = int(parts[1].replace('c', ''))
        latent_dim = int(parts[2].replace('d', ''))
    except Exception as e:
        print(f"Failed to parse VQ-VAE hyperparameters from filename {vae_basename}: {e}")
        num_embeddings = 512
        latent_dim = 32

    print("Loading VQ-VAE weights into memory...")
    try:
        vae_state = torch.load(VAE_PATH, map_location=device, weights_only=True)
        # We can assume standard large model and 64x64 images
        vae = DiscreteVQVAE(latent_dim=latent_dim, model_size='large', input_spatial_dim=64, in_channels=3, num_embeddings=num_embeddings).to(device)
        vae.load_state_dict(vae_state)
        vae.eval()
        print("-> VQ-VAE loaded successfully!")
    except Exception as e:
        print(f"Failed to load VQ-VAE: {e}")
        return

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
    # Extract target latents
    print("Extracting canonical target latents from goal images directly into RAM...")
    target_latents = []
    with torch.no_grad():
        for target in target_normalized:
            img_path = os.path.join(DATA_ROOT, target)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                t_img = transform(img).unsqueeze(0).to(device)
                # Discrete VAE output: recon, quantized, vq_loss, perplexity
                _, quantized, _, _ = vae(t_img)
                target_latents.append(quantized.cpu().squeeze())
            else:
                print(f"WARNING: Target image not found: {img_path}")
                    
    target_latents = torch.stack(target_latents) # [N, latent_dim]
    
    # Calculate the continuous mathematical centroid purely to find the most representative goal image
    centroid_continuous = target_latents.mean(dim=0).cpu().numpy()
    distances_to_continuous = [np.linalg.norm(lat.numpy() - centroid_continuous) for lat in target_latents]
    closest_idx = int(np.argmin(distances_to_continuous))
    closest_image_path = os.path.join(DATA_ROOT, target_normalized[closest_idx])
    
    # 2. Extract EXACT discrete vector for the chosen goal image
    exact_latent_tensor = target_latents[closest_idx]
    exact_latent = exact_latent_tensor.cpu().numpy()
    
    # Expand to the 10 closest topological tokens to form a Group Goal
    with torch.no_grad():
        codebook = vae.vq.embedding.weight.cpu()
        dists = torch.norm(codebook - exact_latent_tensor, dim=1)
        closest_indices = torch.argsort(dists)[:10]
        exact_latents = codebook[closest_indices].cpu().numpy()
        print(f"Expanded Discrete Goal to 10 Tokens. Codebook IDs: {closest_indices.tolist()}")
    
    # Write temp exact stats wrapper file for CQL Dataset loader
    goals_dir = os.path.join(DATA_ROOT, 'goals')
    os.makedirs(goals_dir, exist_ok=True)
    exact_stats_path = os.path.join(goals_dir, 'discrete_exact_stats.json')
    
    stats_data = {
        'exact_latent': exact_latent.tolist(),
        'exact_latents': exact_latents.tolist()
    }
    with open(exact_stats_path, 'w') as f:
        json.dump(stats_data, f)
        
    print(f"Embedded 10-Token Exact Goal Latents to {exact_stats_path}")
        
    # 3. Trigger Offline RL CQL Training Map
    base_model_name = f"{vae_basename}-discrete_cql_model"
    new_model_name = f"{base_model_name}.pth"
    counter = 2
    while os.path.exists(os.path.join(DATA_ROOT, new_model_name)):
        new_model_name = f"{base_model_name}_{counter}.pth"
        counter += 1
        
    print("\n" + "="*50)
    print(f" INITIALIZING DISCRETE CONSERVATIVE Q-LEARNING (CQL) ")
    print("="*50 + "\n")
    
    run_cql_train(
        data_root=DATA_ROOT,
        num_epochs=50,
        vae_model_filename=os.path.basename(VAE_PATH),
        batch_size=128,
        learning_rate=1e-4, 
        alpha=0.2, 
        model_size='large',
        dataset_percent=100,
        goal_type='discrete_exact',
        model_filename=new_model_name,
        train_from_scratch=False
    )
    
    # 4. Duplicate the centroid mapping precisely next to the compiled `.pth` wrapper 
    out_path_full = os.path.join(DATA_ROOT, new_model_name)
    try:
        import shutil
        if os.path.exists(out_path_full):
            centroid_path = out_path_full.replace("model", "centroid").replace(".pth", ".npy")
            np.save(centroid_path, exact_latent)
            print(f"✅ Extracted target Exact Latent wrapper alongside output graph: {os.path.basename(centroid_path)}")
            
            goal_img_out = out_path_full.replace("model", "goal_image").replace(".pth", ".jpg")
            shutil.copy(closest_image_path, goal_img_out)
            print(f"✅ Extracted closest goal image wrapper to {os.path.basename(goal_img_out)}")
    except Exception as e:
        print(f"Failed to extract centroid and goal image configurations: {e}")

if __name__ == "__main__":
    main()
