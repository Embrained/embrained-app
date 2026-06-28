# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training Pipeline for Fixed-Goal CQL using Contrastive Visuomotor Encoder (CVE)
# Supports multiple curated goal images in data/goals/ directory.
# Terminal detection uses VAE nearest-neighbor distance to any goal image.

import os
import sys
import json
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_DIR
from backend.train_cql import train as run_cql_train
from modules.spatial_model import TinyVAE, ContrastiveVisuomotorEncoder

def load_cve_encoder(model_path, device):
    """Load a CVE encoder from a checkpoint path."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(state_dict)
    n_actions = state_dict['action_predictor.2.weight'].shape[0]
    encoder = ContrastiveVisuomotorEncoder(
        latent_dim=latent_dim, model_size=model_size,
        input_spatial_dim=img_dim, in_channels=in_channels,
        n_actions=n_actions
    ).to(device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder, latent_dim, img_dim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting CVE Fixed-Goal CQL Offline Reinforcement Learning on {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    # Load all goal images from data/goals/
    goals_dir = os.path.join(DATA_ROOT, 'goals')
    goal_images = sorted(glob.glob(os.path.join(goals_dir, '*.jpg')))
    if not goal_images:
        print("ERROR: No goal images found in data/goals/. Place curated goal images there.")
        return
    
    print(f"Found {len(goal_images)} goal images in data/goals/")
        
    # Find the latest CVE model
    print("Searching for latest CVE checkpoint in data directory...")
    cve_candidates = glob.glob(os.path.join(DATA_ROOT, '*cve*.pth'))
    cve_candidates = [f for f in cve_candidates if not any(x in f.lower() for x in ['hello_world', 'cql', 'reflex', 'fixed_goal', 'policy'])]
    if not cve_candidates:
        print("Error: No CVE model found matching pattern '*cve*.pth'!")
        return
        
    cve_candidates.sort(key=os.path.getmtime, reverse=True)
    CVE_PATH = cve_candidates[0]
    cve_basename = os.path.basename(CVE_PATH).replace('.pth', '')
    print(f"-> Selected latest CVE: {os.path.basename(CVE_PATH)}")

    print("Loading CVE encoder weights into memory...")
    try:
        encoder, latent_dim, img_dim = load_cve_encoder(CVE_PATH, device)
        print(f"-> CVE loaded successfully! (Latent: {latent_dim}d, Input: {img_dim}x{img_dim})")
    except Exception as e:
        print(f"Failed to load CVE: {e}")
        return

    transform = T.Compose([T.Resize((img_dim, img_dim)), T.ToTensor()])
    
    # Encode all goal images with CVE
    print(f"Encoding {len(goal_images)} goal images via CVE encoder...")
    goal_latents = []
    with torch.no_grad():
        for gp in goal_images:
            img = Image.open(gp).convert('RGB')
            t_img = transform(img).unsqueeze(0).to(device)
            mu = encoder.encode(t_img).cpu().squeeze().numpy()
            goal_latents.append(mu)
            
    goal_latents = np.array(goal_latents)  # [N, 32]
    centroid = np.mean(goal_latents, axis=0)
    
    # Stats
    dists_to_centroid = np.linalg.norm(goal_latents - centroid, axis=1)
    print(f"Goal centroid computed from {len(goal_latents)} images")
    print(f"  Avg dist to centroid: {np.mean(dists_to_centroid):.4f}")
    print(f"  Max dist to centroid: {np.max(dists_to_centroid):.4f}")
    
    # Save group stats for CQL Dataset loader
    group_stats_path = os.path.join(goals_dir, 'group_stats.json')
    stats_data = {
        'centroid': centroid.tolist(),
        'goal_latents': goal_latents.tolist(),
        'num_goals': len(goal_images),
        'goal_images': [os.path.basename(g) for g in goal_images]
    }
    with open(group_stats_path, 'w') as f:
        json.dump(stats_data, f)
        
    print(f"Saved goal stats ({len(goal_images)} goals + centroid) to {group_stats_path}")
        
    # Trigger Offline RL CQL Training
    base_model_name = f"{cve_basename}-fixed_goal_cql_model"
    new_model_name = f"{base_model_name}.pth"
    counter = 2
    while os.path.exists(os.path.join(DATA_ROOT, new_model_name)):
        new_model_name = f"{base_model_name}_{counter}.pth"
        counter += 1
        
    print("\n" + "="*50)
    print(" INITIALIZING CVE CONSERVATIVE Q-LEARNING (CQL) ")
    print("="*50 + "\n")
    
    run_cql_train(
        data_root=DATA_ROOT,
        num_epochs=300,
        vae_model_filename=os.path.basename(CVE_PATH),
        batch_size=128,
        learning_rate=5e-5,
        alpha=0.1,
        model_size='medium',
        dataset_percent=100,
        goal_type='group_goal',
        model_filename=new_model_name,
        train_from_scratch=False
    )
    
    # Save sidecar files next to the compiled .pth
    out_path_full = os.path.join(DATA_ROOT, new_model_name)
    try:
        import shutil
        if os.path.exists(out_path_full):
            # Save centroid
            centroid_path = out_path_full.replace("model", "centroid").replace(".pth", ".npy")
            np.save(centroid_path, centroid)
            print(f"✅ Saved goal centroid: {os.path.basename(centroid_path)}")
            
            # Save the goal image closest to centroid as the representative
            closest_idx = int(np.argmin(dists_to_centroid))
            goal_img_out = out_path_full.replace("model", "goal_image").replace(".pth", ".jpg")
            shutil.copy(goal_images[closest_idx], goal_img_out)
            print(f"✅ Saved representative goal image: {os.path.basename(goal_img_out)}")
    except Exception as e:
        print(f"Failed to save sidecar files: {e}")

if __name__ == "__main__":
    main()
