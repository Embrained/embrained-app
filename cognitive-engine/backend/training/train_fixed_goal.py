# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training Pipeline for Hello World: Fixed-Goal Behavioral Cloning

import os
import sys
import json
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import ACTION_PWM_MAP, MODELS_DIR, DATA_DIR
from modules.spatial_model import TinyVAE

T_HORIZON = 5 # Number of steps preceding the goal image to count as "successful trajectory"

TARGET_IMAGES = []
goals_dir = os.path.join(DATA_DIR, 'goals')
if os.path.exists(goals_dir):
    for f in os.listdir(goals_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            TARGET_IMAGES.append(f"goals/{f}")

if not TARGET_IMAGES:
    print("Warning: No images found in data/goals. Falling back to hardcoded goals.")
    TARGET_IMAGES = [
        "markov_2026-03-22_13-34-29/images/frame_1774201370056.jpg",
        "markov_2026-03-23_17-13-37/images/frame_1774300483517.jpg",
        "markov_2026-03-23_17-13-37/images/frame_1774300685051.jpg",
        "markov_2026-03-28_15-36-36/images/frame_1774726629342.jpg",
        "markov_2026-03-28_15-36-36/images/frame_1774727221978.jpg",
        "markov_2026-04-09_17-57-21/images/frame_1775771989581.jpg",
        "markov_2026-04-09_19-01-19/images/frame_1775775762862.jpg",
        "markov_2026-04-10_10-03-48/images/frame_1775829945267.jpg",
        "markov_2026-04-10_10-03-48/images/frame_1775830001252.jpg",
        "markov_2026-04-10_10-03-48/images/frame_1775830792618.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776300989512.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776301260549.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776301558193.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776301789015.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776298380051.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776298778840.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776298791319.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776299291559.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776299387708.jpg",
        "markov_2026-04-15_19-52-06/images/frame_1776297443526.jpg",
        "markov_2026-04-15_19-52-06/images/frame_1776297814967.jpg",
        "markov_2026-04-15_19-52-06/images/frame_1776298091052.jpg",
        "markov_2026-04-15_17-17-37/images/frame_1776288722567.jpg",
        "markov_2026-04-15_17-17-37/images/frame_1776289155128.jpg",
        "markov_2026-04-15_17-17-37/images/frame_1776289567346.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287001218.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287067469.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287320412.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287321847.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285752462.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285787367.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285794981.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285807256.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776286083024.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283828658.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283855887.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283954649.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283949674.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776284138608.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776284184794.jpg"
    ]

class FixedGoalBCNetwork(nn.Module):
    """
    Acts like an OracleQNetwork so we can load it directly into the Planner.
    We just set goal_dim=0 because the policy learns unconditionally to go to the fixed goal.
    """
    def __init__(self, state_dim=32, action_dim=5):
        super().__init__()
        # To match the planner's introspection logic where input_layer in_features == state_dim
        # Actually in planner.py, it expects state_dim + goal_dim if it's CQL, but OracleQNetwork
        # takes state_dim + goal_dim. If we want it to just take state, we use an OracleQNetwork structure.
        
        # Planner passes obs: [z_stacked] which is 96 dims if pure reflex OR [z_stacked, z_goal]
        # Wait, if planner sees 'dark_wall' it sets is_pure_reflex. We'll name our model:
        # fixed_goal_dark_wall... to trigger reflex or just implement it normally.
        
        # Input: 3 temporally stacked latents
        input_dim = state_dim * 3
        
        self.input_layer = nn.Linear(input_dim, 256)
        
        self.net = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.use_ln = False
        
    def forward(self, state, goal=None):
        return self.net(state)


class BCDataset(Dataset):
    def __init__(self, transitions, latent_dict):
        self.transitions = transitions
        self.latent_dict = latent_dict

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        item = self.transitions[idx]
        
        # Frame stacking: we need the current and 2 previous latents
        # Since we have isolated transitions right now, let's just pad if we don't have history
        # Or even better, just pass the same latent 3 times for simplicity on this Hello World model
        
        p = item.get('image_path', '')
        if p in self.latent_dict:
            latent = self.latent_dict[p]
        else:
            sample_latent = next(iter(self.latent_dict.values()))
            latent = torch.zeros_like(sample_latent)
        
        # Stacked: 3 copies of latent
        stacked_latent = torch.cat([latent, latent, latent], dim=0)

        # Action dubins mapping
        best_action = -1
        if 'macro_action' in item:
            best_action = int(item['macro_action'])
        else:
            raw_l = float(item.get('left_cmd', 0.0))
            raw_r = float(item.get('right_cmd', 0.0))
            best_dist = float('inf')
            for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
                dist = math.hypot(raw_l - map_l, raw_r - map_r)
                if dist < best_dist:
                    best_dist = dist
                    best_action = act_id
        
        # Map to valid action embedding
        # Dubins actions with Reverse restored: Forward=1, Reverse=2, Left=3, Right=4
        if best_action == 1: act_idx = 0
        elif best_action == 2: act_idx = 1
        elif best_action == 3: act_idx = 2
        elif best_action == 4: act_idx = 3
        else: act_idx = 0 # Default to forward if invalid

        return stacked_latent, torch.tensor(act_idx, dtype=torch.long)

def normalize_path(p):
    """Normalize path strings for comparison."""
    if not p: return ""
    return str(p).replace('/', '\\').split('data\\')[-1]

def train(data_root="data", epochs=50, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Fixed-Goal Behavioral Cloning on {device}")
    
    DATA_ROOT = os.path.abspath(data_root)
    
    # Load Transitions
    trans_path = os.path.join(DATA_ROOT, "all_transitions.json")
    if not os.path.exists(trans_path):
         print(f"Error: {trans_path} not found.")
         return
         
    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    # Group by session
    sessions = {}
    for item in all_data:
        s = item['session']
        if s not in sessions: sessions[s] = []
        sessions[s].append(item)
        
    target_normalized = [normalize_path(p) for p in TARGET_IMAGES]
        
    # Dynamically find the latest VAE model
    import glob
    print("Searching for latest VAE checkpoint in data directory...")
    vae_candidates = glob.glob(os.path.join(DATA_ROOT, '*-vae_*.pth'))
    # Filter out generated policy files that accidentally match the prefix
    vae_candidates = [f for f in vae_candidates if not any(x in f.lower() for x in ['hello_world', 'cql', 'reflex', 'fixed_goal', 'policy'])]
    if not vae_candidates:
        print("Error: No VAE found matching pattern '*-vae_*.pth'!")
        return
        
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    VAE_PATH = vae_candidates[0]
    print(f"-> Selected latest VAE: {os.path.basename(VAE_PATH)}")

    print("Loading VAE weights into memory...")
    try:
        vae_state = torch.load(VAE_PATH, map_location=device, weights_only=True)
        latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(vae_state)
        vae = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=img_dim, in_channels=in_channels).to(device)
        vae.load_state_dict(vae_state)
        vae.eval()
        print("-> VAE loaded successfully!")
    except Exception as e:
        print(f"Failed to load VAE: {e}")
        return

    # Set up cache_path
    vae_basename = os.path.basename(VAE_PATH).replace('.pth', '')
    cache_path = os.path.join(DATA_ROOT, f"{vae_basename}_global_latents.pt")
    print(f"Targeting latent cache: {os.path.basename(cache_path)}")
        
    # Re-index dict with standardized backslashes
    latent_dict = {}
    if os.path.exists(cache_path):
        print("-> Cache found! Loading latents into memory...")
        raw_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("path_map", {})
        for k, v in raw_dict.items():
            latent_dict[normalize_path(k)] = v
        print(f"-> Pre-loaded {len(latent_dict)} hashed paths from cache.")
    else:
        print("-> No existing latent cache found. On-the-fly extraction will be used (this may take a while).")
        
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
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
                    _, mu, _ = vae(t_img)
                    target_latents.append(mu.cpu().squeeze())
                    latent_dict[target] = mu.cpu().squeeze()
                else:
                    print(f"WARNING: Target image not found: {img_path}")
                    
    target_latents = torch.stack(target_latents) # [10, 32]
    
    expert_transitions = []
    print("Isolating trajectories ending within targeted latent tolerance...")
    
    for s_name, traj in sessions.items():
        traj = sorted(traj, key=lambda x: x['timestamp'])
        
        # Find if target distance is met in this session
        target_indices = []
        for i, node in enumerate(traj):
            p = normalize_path(node.get('image_path', ''))
            
            # Fast ad-hoc extraction for missing latents
            if p and p not in latent_dict:
                img_path = os.path.join(DATA_ROOT, p)
                if os.path.exists(img_path):
                    if len(latent_dict) % 500 == 0:
                        print(f"   [Latent Cache Miss] On-the-fly computing latent for frame: {p}...")
                    img = Image.open(img_path).convert('RGB')
                    t_img = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, mu, _ = vae(t_img)
                    latent_dict[p] = mu.cpu().squeeze()
            
            if p in latent_dict:
                z = latent_dict[p]
                dists = torch.norm(target_latents - z.unsqueeze(0), dim=1)
                min_dist = torch.min(dists).item()
                if min_dist <= 1.5:
                    target_indices.append(i)
                
        # If target found, extract the T_HORIZON previous steps
        for t_idx in target_indices:
            start_idx = max(0, t_idx - T_HORIZON)
            for j in range(start_idx, t_idx):
                # Don't add stops (0)
                act = traj[j].get('macro_action', -1)
                if act in [1, 2, 3, 4]: 
                    expert_transitions.append(traj[j])

    # Remove duplicates because multiple target_indices could overlap their horizons
    expert_transitions = {id(n): n for n in expert_transitions}.values()
    expert_transitions = list(expert_transitions)

    print(f"Identified {len(expert_transitions)} expert transitions leading to the target goal.")
    
    if len(expert_transitions) == 0:
        print("No expert transitions found! Increase tolerance or check target list.")
        return


    # Train Policy (4 Actions: Fwd, Rev, Left, Right)
    model = FixedGoalBCNetwork(state_dim=latent_dim, action_dim=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = BCDataset(expert_transitions, latent_dict)
    # Even if dataset is small, dataloader drops last if drop_last=True. 
    # Use smaller batch size appropriately
    print(f"\nFinal dataset compiled: {len(dataset)} examples. Initializing dataloader.")
    actual_batch = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=actual_batch, shuffle=True)
    
    print("\n--- Training Initialization ---")
    print(f"Model Structure: {latent_dim*3}-Dimensional Stacked Input -> MLP -> 4 Actions")
    print(f"Batch Size: {actual_batch} | Learning Rate: {lr} | Epochs: {epochs}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for latents, actions in dataloader:
            latents, actions = latents.to(device), actions.to(device)
            
            logits = model(latents)
            loss = criterion(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == actions).sum().item()
            total += actions.size(0)
            
        acc = 100.0 * correct / total
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.1f}%")

    # Pack state dict nicely for Planner.py
    # We pretend it's CQL compatible (with action_dim=5 padding if needed, but Planner dynamically matches dimensions if we are careful)
    # Actually wait, config.ACTION_DIM is 3 because the user constrained it recently!
    # "Standardizing Neural Navigation Oracles... action space (Forward, Left, Right)"
    
    out_path = os.path.join(DATA_ROOT, f"{vae_basename}-fixed_goal_model.pth")
    torch.save(model.state_dict(), out_path)
    
    # Save the aggregated centroid for use by the Live Planner's group-goal abstraction
    try:
        centroid_path = out_path.replace("_model.pth", "_centroid.npy")
        centroid = target_latents.mean(dim=0).cpu().numpy()
        np.save(centroid_path, centroid)
        print(f"Goal Centroid envelope generated and saved to {centroid_path}")
    except Exception as e:
        print(f"Failed to save Centroid envelope: {e}")
        
    print(f"\nFixed Goal Policy compiled successfully! Saved to {out_path}")


if __name__ == "__main__":
    # Typically parent dir of backend is software_suite
    train(data_root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
