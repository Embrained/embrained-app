# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training Pipeline for Goal-Conditioned CQL using Contrastive Visuomotor Encoder (CVE)
# Uses HER (Hindsight Experience Replay) for flexible goal conditioning.

import os
import sys
import glob
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_DIR
from backend.train_cql import train as run_cql_train
from modules.spatial_model import TinyVAE, ContrastiveVisuomotorEncoder


def find_latest_cve(data_root):
    """Find the most recent CVE checkpoint in the data directory."""
    candidates = glob.glob(os.path.join(data_root, '*cve*.pth'))
    # Filter out policy/CQL files
    candidates = [f for f in candidates if not any(x in f.lower() for x in [
        'hello_world', 'cql', 'reflex', 'fixed_goal', 'policy', 'goal_conditioned'
    ])]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting CVE Goal-Conditioned CQL (HER) on {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    # Find latest CVE model
    print("Searching for latest CVE checkpoint in data directory...")
    CVE_PATH = find_latest_cve(DATA_ROOT)
    if not CVE_PATH:
        print("Error: No CVE model found matching pattern '*cve*.pth' or '*vae_contrastive*.pth'!")
        return
    
    cve_basename = os.path.basename(CVE_PATH).replace('.pth', '')
    print(f"-> Selected latest CVE: {os.path.basename(CVE_PATH)}")
    
    # Verify it loads correctly
    try:
        state_dict = torch.load(CVE_PATH, map_location=device, weights_only=True)
        latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(state_dict)
        assert 'action_predictor.0.weight' in state_dict, "Not a CVE model!"
        print(f"-> CVE verified: {latent_dim}d latent, {model_size} encoder, {img_dim}x{img_dim} input")
    except Exception as e:
        print(f"Failed to verify CVE model: {e}")
        return
    
    # Build output filename
    base_model_name = f"{cve_basename}-goal_conditioned_cql_model"
    new_model_name = f"{base_model_name}.pth"
    counter = 2
    while os.path.exists(os.path.join(DATA_ROOT, new_model_name)):
        new_model_name = f"{base_model_name}_{counter}.pth"
        counter += 1
        
    print("\n" + "="*60)
    print(f"[CVE] GOAL-CONDITIONED CQL (Hindsight Experience Replay)")
    print("="*60 + "\n")
    
    run_cql_train(
        data_root=DATA_ROOT,
        num_epochs=50,
        vae_model_filename=os.path.basename(CVE_PATH),
        batch_size=128,
        learning_rate=1e-4,
        alpha=0.2,
        model_size='large',
        dataset_percent=100,
        goal_type='her',
        model_filename=new_model_name,
        train_from_scratch=False
    )
    
    out_path_full = os.path.join(DATA_ROOT, new_model_name)
    if os.path.exists(out_path_full):
        print(f"\n[OK] Goal-conditioned CQL policy saved to: {os.path.basename(out_path_full)}")
    else:
        print(f"\n[WARN] Expected output not found at: {out_path_full}")

if __name__ == "__main__":
    main()
