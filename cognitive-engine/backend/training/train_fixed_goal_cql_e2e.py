# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training Pipeline for Hello World: Fixed-Goal Offline Reinforcement Learning (CQL) End-to-End

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_DIR
from backend.training.train_fixed_goal import TARGET_IMAGES, normalize_path
from backend.train_cql import train as run_cql_train
from modules.spatial_model import TinyVAE, ContrastiveVisuomotorEncoder

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting End-to-End Fixed-Goal CQL Offline Reinforcement Learning on {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    target_normalized = [normalize_path(p) for p in TARGET_IMAGES]
        
    # Dynamically find the latest CVE model
    import glob
    print("Searching for latest CVE checkpoint in data directory...")
    cve_candidates = glob.glob(os.path.join(DATA_ROOT, 'cve_*.pth'))
    # Filter out generated policy files that accidentally match the prefix
    cve_candidates = [f for f in cve_candidates if not any(x in f.lower() for x in ['hello_world', 'cql', 'reflex', 'fixed_goal', 'policy'])]
    if not cve_candidates:
        print("Error: No CVE found matching pattern 'cve_*.pth'!")
        return
        
    cve_candidates.sort(key=os.path.getmtime, reverse=True)
    CVE_PATH = cve_candidates[0]
    cve_basename = os.path.basename(CVE_PATH).replace('.pth', '')
    print(f"-> Selected latest CVE: {os.path.basename(CVE_PATH)}")

    print("Loading CVE weights into memory...")
    try:
        cve_state = torch.load(CVE_PATH, map_location=device, weights_only=True)
        if "action_predictor.0.weight" not in cve_state:
            print("Error: Model state dict does not appear to be a CVE.")
            return
        
        latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(cve_state)
        n_actions = cve_state['action_predictor.2.weight'].shape[0]
        
        cve = ContrastiveVisuomotorEncoder(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=img_dim, in_channels=in_channels, n_actions=n_actions).to(device)
        cve.load_state_dict(cve_state)
        cve.eval()
        print(f"-> CVE loaded successfully! ({latent_dim}d latent, {model_size} encoder, {img_dim}x{img_dim})")
    except Exception as e:
        print(f"Failed to load CVE: {e}")
        return

    # Set up cache_path
    cache_path = os.path.join(DATA_ROOT, f"{cve_basename}_global_latents.pt")
    print(f"Targeting latent cache: {os.path.basename(cache_path)}")
        
    latent_dict = {}
    if os.path.exists(cache_path):
        print("-> Cache found! Loading latents into memory...")
        raw_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("path_map", {})
        for k, v in raw_dict.items():
            latent_dict[normalize_path(k)] = v
    else:
        print("-> No existing latent cache found. On-the-fly extraction will be used.")
        
    transform = T.Compose([T.Resize((img_dim, img_dim)), T.ToTensor()])
    
    # Extract target latents
    print("Extracting canonical target latents from goal images...")
    target_latents = []
    with torch.no_grad():
        for target in target_normalized:
            if target in latent_dict:
                target_latents.append(latent_dict[target])
            else:
                img_path = os.path.join(DATA_ROOT, target)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    t_img = transform(img).unsqueeze(0).to(device)
                    mu = cve.encode(t_img)
                    target_latents.append(mu.cpu().squeeze())
                    latent_dict[target] = mu.cpu().squeeze()
                else:
                    print(f"WARNING: Target image not found: {img_path}")
                    
    target_latents = torch.stack(target_latents) # [10, 32]
    centroid = target_latents.mean(dim=0).cpu().numpy()
    
    # 2. Write temp group stats wrapper file for CQL Dataset loader
    goals_dir = os.path.join(DATA_ROOT, 'goals')
    os.makedirs(goals_dir, exist_ok=True)
    group_stats_path = os.path.join(goals_dir, 'group_stats.json')
    
    stats_data = {
        'centroid': centroid.tolist(),
        'average_in_group_distance': 1.5  # Force explicit 1.5 cutoff to match standard BC training cutoff
    }
    with open(group_stats_path, 'w') as f:
        json.dump(stats_data, f)
        
    print(f"Embedded Centroid target bounds to {group_stats_path}")
        
    # 3. Trigger Offline RL CQL Training Map
    new_model_name = f"{cve_basename}-fixed_goal_cql_e2e_model.pth"
    print("\n" + "="*50)
    print(f"🔥 INITIALIZING END-TO-END CONSERVATIVE Q-LEARNING (CQL) 🔥")
    print("="*50 + "\n")
    
    run_cql_train(
        data_root=DATA_ROOT,
        num_epochs=50,
        vae_model_filename=os.path.basename(CVE_PATH),
        batch_size=64, # Reduced from 128 to combat OOM issues during end-to-end vision backpropagation
        learning_rate=1e-4, # Standard CQL learning rate
        model_size='large',
        dataset_percent=100,
        goal_type='group_goal',
        model_filename=new_model_name,
        train_from_scratch=True # ENABLED backpropagation through the visual encoder
    )
    
    # 4. Duplicate the centroid mapping precisely next to the compiled `.pth` wrapper 
    # to structurally signal the Planner it's an explicit coordinate bounds model!
    out_path_full = os.path.join(DATA_ROOT, new_model_name)
    try:
        if os.path.exists(out_path_full):
            centroid_path = out_path_full.replace("_model.pth", "_centroid.npy")
            np.save(centroid_path, centroid)
            print(f"✅ Extracted target Centroid bound wrapper alongside output graph: {os.path.basename(centroid_path)}")
    except Exception as e:
        print(f"Failed to copy standard centroid envelope configuration correctly into {DATA_ROOT}: {e}")

if __name__ == "__main__":
    main()
