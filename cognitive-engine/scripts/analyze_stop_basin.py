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
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torchvision import transforms

# --- CONFIGURATION ---
DATA_ROOT = r"C:\Users\chris\Embrained\embrained-app\data\nook"
JSON_FILE = "all_transitions.json"
VAE_MODEL = "nook-vae.pth"
CQL_MODEL = "nook-vae-cql.pth"
IMG_H, IMG_W = 64, 64

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from modules.spatial_model import TinyVAE, CQLNetwork

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

def load_data(root, json_name):
    path = os.path.join(root, json_name)
    print(f"Loading dataset from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    # Filter for valid images
    valid_data = []
    print("Verifying image paths...")
    for idx, item in enumerate(data):
        img_path = os.path.join(root, item['image_path'])
        if os.path.exists(img_path):
             valid_data.append(item)
    print(f"Loaded {len(valid_data)} valid transitions.")
    return valid_data

def load_models(root):
    # 1. Load VAE
    vae_path = os.path.join(root, VAE_MODEL)
    print(f"Loading VAE from {vae_path}...")
    vae = TinyVAE(latent_dim=32, model_size='small').to(device) # Default
    
    detected_size = 'small'
    
    # Auto-detect size if possible
    try:
        state = torch.load(vae_path, map_location=device)
        
        # Detect dimensions
        if 'fc_mu.weight' in state:
             ld = state['fc_mu.weight'].shape[0]
             fd = state['fc_mu.weight'].shape[1]
             print(f"Detected VAE Latent={ld}, Flatten={fd}")
             
             # Re-init if mismatch from default
             # Heuristic based on spatial_model.py
             if fd == 2048: detected_size = 'large'
             elif fd == 8192: detected_size = 'medium'
             elif fd == 4096: detected_size = 'tiny' 
             # Note: 'small' is ambiguous with medium/tiny depending on base channels but usually 4096 or 8192?
             # Let's trust the flatten dim for now.
             
             vae = TinyVAE(latent_dim=ld, model_size=detected_size).to(device)
                  
        if 'encoder.0.weight' in state:
             vae.load_state_dict(state, strict=False)
        else:
             vae.encoder.load_state_dict(state)
             
    except Exception as e:
        print(f"Error loading VAE: {e}")
        sys.exit(1)
        
    vae.eval()
    
    # 2. Load Policy
    policy_path = os.path.join(root, CQL_MODEL)
    print(f"Loading Policy from {policy_path} (Using detected size: {detected_size})...")
    
    # Policy input is 2 * latent
    # Note: CQLNetwork args are (input_dim, hidden_dim, action_dim, use_ln, model_size)
    # But hidden_dim is OVERRIDDEN by model_size inside __init__! 
    # So we just need to pass model_size correctly.
    
    policy = CQLNetwork(input_dim=vae.latent_dim * 2, action_dim=5, model_size=detected_size).to(device)
    
    try:
        p_state = torch.load(policy_path, map_location=device)
        if 'model_state_dict' in p_state:
             policy.load_state_dict(p_state['model_state_dict'])
        else:
             policy.load_state_dict(p_state)
    except Exception as e:
         print(f"Error loading Policy: {e}")
         sys.exit(1)
         
    policy.eval()
    
    return vae, policy

def extract_latents(vae, data, root):
    print("Extracting latents (this may take a minute)...")
    latents = []
    
    # Batch processing for speed? Or just simple loop
    batch_size = 64
    images = []
    
    with torch.no_grad():
        for i, item in enumerate(data):
            img_path = os.path.join(root, item['image_path'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t_img = transform(img)
            images.append(t_img)
            
            if len(images) >= batch_size or i == len(data) - 1:
                batch = torch.stack(images).to(device)
                _, mu, _ = vae(batch)
                latents.append(mu.cpu().numpy())
                images = []
                
    return np.concatenate(latents, axis=0)

def main():
    # 1. Setup
    data = load_data(DATA_ROOT, JSON_FILE)
    vae, policy = load_models(DATA_ROOT)
    
    # 2. Phase Mapping
    latents = extract_latents(vae, data, DATA_ROOT)
    
    print("Computing PCA and Phase...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latents)
    
    # Phase calculation: [-pi, pi]
    phases = np.arctan2(z_pca[:, 1], z_pca[:, 0]) 
    phases_deg = np.degrees(phases) # [-180, 180]
    
    # 3. Virtual Sweep Setup
    # Pick a random goal index that isn't at the very edge to avoid wrap-around headaches for now
    # Or just handle wrap around. 
    # Let's just pick one.
    
    goal_idx = np.random.randint(0, len(data))
    goal_z = latents[goal_idx]
    goal_phase = phases_deg[goal_idx]
    
    print(f"Selected Goal Index: {goal_idx}, Phase: {goal_phase:.2f} deg")
    
    # Find sweep images: +/- 45 degrees
    # Handle wrap-around diff
    diffs = phases_deg - goal_phase
    diffs = (diffs + 180) % 360 - 180 # Normalize to [-180, 180]
    
    mask = (np.abs(diffs) <= 45.0)
    sweep_indices = np.where(mask)[0]
    
    if len(sweep_indices) < 10:
         print("Not enough samples in sweep range. Retrying...")
         return # Naive
         
    # Sort by signed error (negative = behind/left?, positive = ahead/right?)
    # "Left" vs "Right" depends on topology direction but usually consistent.
    sorted_sweep_args = np.argsort(diffs[sweep_indices])
    sweep_indices = sweep_indices[sorted_sweep_args]
    
    sweep_errors = diffs[sweep_indices]
    sweep_latents = latents[sweep_indices]
    
    print(f"Sweep contains {len(sweep_indices)} frames from {sweep_errors[0]:.1f} to {sweep_errors[-1]:.1f} deg.")
    
    # 4. Inference
    print("Running Policy Inference...")
    
    probs_left = []
    probs_right = []
    probs_stop = []
    
    goal_tensor = torch.FloatTensor(goal_z).unsqueeze(0).to(device) # (1, 32)
    
    with torch.no_grad():
        for i in range(len(sweep_indices)):
            curr_z = torch.FloatTensor(sweep_latents[i]).unsqueeze(0).to(device)
            
            # Input: cat(curr, goal)
            state = torch.cat([curr_z, goal_tensor], dim=1)
            
            logits = policy(state) # (1, 5)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # 0:Fwd, 1:Left, 2:Right, 3:Stop, 4:Back
            # Left (Green), Right (Blue), Stop (Red)
            # Fwd usually adds to both or centers. 
            # Prompt asks for Left/Right/Stop curves.
            
            probs_left.append(probs[1])
            probs_right.append(probs[2])
            probs_stop.append(probs[3])
            
    # 5. Visualization (The Red Mountain)
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    plt.plot(sweep_errors, probs_left, c='lime', label='Left (1)', linewidth=2, alpha=0.8)
    plt.plot(sweep_errors, probs_right, c='cyan', label='Right (2)', linewidth=2, alpha=0.8)
    plt.plot(sweep_errors, probs_stop, c='red', label='Stop (3)', linewidth=3)
    
    plt.axvline(0, color='white', linestyle='--', alpha=0.5, label='Goal')
    
    # Deadband region markings
    plt.axvspan(-5, 5, color='red', alpha=0.1, label='Target Stop Basin')
    
    plt.title(f"Policy Decision Boundary (Goal Phase: {goal_phase:.1f}$^\circ$)", fontsize=14)
    plt.xlabel("Angular Phase Error (degrees)", fontsize=12)
    plt.ylabel("Action Probability", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.ylim(0, 1.05)
    
    out_path = "stop_basin_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    # plt.show() # Can't show in headless
    
if __name__ == "__main__":
    main()
