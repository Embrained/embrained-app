# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training Pipeline for Seek CQL using classifier-shaped rewards.
# The goal is implicit in the reward function: the robot learns to navigate
# toward goal close-up views, using a pre-trained GoalClassifier as the
# reward signal.
#
# CQL input is 32-dim CVE latent only (no goal concatenation).

import os
import sys
import glob
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cognitive-engine')))
from config import DATA_DIR
from backend.train_cql import train as run_cql_train
from modules.spatial_model import TinyVAE, ContrastiveVisuomotorEncoder
from modules.goal_classifier import GoalClassifier


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f" SEEK CQL TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    # --- 1. Goal Classifier ---
    goal_subdir = 'sofa'  # Default target
    if len(sys.argv) > 1:
        goal_subdir = sys.argv[1]
    
    classifier_path = os.path.join(DATA_ROOT, f'{goal_subdir}_classifier.pth')
    if not os.path.exists(classifier_path):
        print(f"ERROR: Classifier not found at {classifier_path}")
        print(f"Run train_classifier.py first to train the {goal_subdir} classifier.")
        return
    
    print(f"[OK] Found {goal_subdir} classifier at {classifier_path}")
    
    # Verify classifier loads
    classifier = GoalClassifier.load_from_checkpoint(classifier_path, device=str(device))
    print(f"   Classifier loaded successfully ({sum(p.numel() for p in classifier.parameters()):,} params)")
    del classifier  # CQL training will load it internally
    
    # --- 2. CVE Encoder ---
    print("\nSearching for latest CVE checkpoint in data directory...")
    cve_candidates = glob.glob(os.path.join(DATA_ROOT, 'cve_*.pth'))
    cve_candidates = [f for f in cve_candidates if not any(x in f.lower() for x in ['cql', 'policy', 'classifier'])]
    
    if not cve_candidates:
        print("ERROR: No CVE model found matching pattern 'cve_*.pth'!")
        return
    
    cve_candidates.sort(key=os.path.getmtime, reverse=True)
    CVE_PATH = cve_candidates[0]
    cve_basename = os.path.basename(CVE_PATH).replace('.pth', '')
    print(f"-> Selected latest CVE: {os.path.basename(CVE_PATH)}")
    
    # Verify CVE loads
    try:
        state_dict = torch.load(CVE_PATH, map_location=device, weights_only=True)
        latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(state_dict)
        print(f"   CVE: latent_dim={latent_dim}, model_size={model_size}, img_dim={img_dim}")
        del state_dict
    except Exception as e:
        print(f"ERROR: Failed to load CVE: {e}")
        return
    
    # --- 3. Generate output name ---
    base_model_name = f"{cve_basename}-{goal_subdir}_seek_cql_model"
    new_model_name = f"{base_model_name}.pth"
    counter = 2
    while os.path.exists(os.path.join(DATA_ROOT, new_model_name)):
        new_model_name = f"{base_model_name}_{counter}.pth"
        counter += 1
    
    print(f"\nOutput model: {new_model_name}")
    
    # --- 4. Run CQL Training ---
    print(f"\n{'='*50}")
    print(f" INITIALIZING CONSERVATIVE Q-LEARNING (CQL)")
    print(f" Goal Type: {goal_subdir}_seek")
    print(f" CQL Input: {latent_dim}-dim (no goal concatenation)")
    print(f"{'='*50}\n")
    
    run_cql_train(
        data_root=DATA_ROOT,
        num_epochs=200,
        vae_model_filename=os.path.basename(CVE_PATH),
        batch_size=128,
        learning_rate=5e-5,
        alpha=0.1,
        model_size='medium',
        dataset_percent=100,
        goal_type=f'{goal_subdir}_seek',
        model_filename=new_model_name,
        train_from_scratch=False,
    )
    
    print(f"\n[OK] {goal_subdir.capitalize()}-Seeking CQL training complete!")
    print(f"   Model saved to: {os.path.join(DATA_ROOT, new_model_name)}")


if __name__ == '__main__':
    main()
