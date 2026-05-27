# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training Pipeline for Hello World: IR Reflex Wall-Avoidance

import os
import sys
import json
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import ACTION_PWM_MAP, MODELS_DIR
from modules.spatial_model import TinyVAE, IRPredictorNetwork

class IRDataset(Dataset):
    def __init__(self, transitions, latent_dict):
        self.transitions = []
        # Filter for transitions that have sonar data and a valid next state
        for i in range(len(transitions) - 1):
            curr = transitions[i]
            nxt = transitions[i+1]
            
            # Need to be in the same session
            if curr['session'] != nxt['session']:
                 continue
                 
            # Need distance reading
            if 'dist' not in nxt:
                 continue
                 
            self.transitions.append((curr, nxt))
            
        self.latent_dict = latent_dict

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        curr, nxt = self.transitions[idx]
        
        p = curr.get('image_path', '')
        if p in self.latent_dict:
            latent = self.latent_dict[p]
        else:
            sample_latent = next(iter(self.latent_dict.values()))
            latent = torch.zeros_like(sample_latent)
        
        # Stacked: 3 copies of latent for simple 96-dim input
        stacked_latent = torch.cat([latent, latent, latent], dim=0)

        # Action dubins mapping
        best_action = -1
        if 'macro_action' in curr:
            best_action = int(curr['macro_action'])
        else:
            raw_l = float(curr.get('left_cmd', 0.0))
            raw_r = float(curr.get('right_cmd', 0.0))
            best_dist = float('inf')
            for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
                dist = math.hypot(raw_l - map_l, raw_r - map_r)
                if dist < best_dist:
                    best_dist = dist
                    best_action = act_id
        
        # Map to valid action embedding
        if best_action == 1: act_idx = 0
        elif best_action == 3: act_idx = 1
        elif best_action == 4: act_idx = 2
        else: act_idx = 0 

        next_sonar = float(nxt['dist'])
        
        return stacked_latent, torch.tensor(act_idx, dtype=torch.long), torch.tensor([next_sonar], dtype=torch.float32)

def train(data_root="data", epochs=200, batch_size=128, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting IR Reflex Forward Model Training on {device}")
    
    DATA_ROOT = os.path.abspath(data_root)
    
    # Load Transitions
    trans_path = os.path.join(DATA_ROOT, "all_transitions.json")
    if not os.path.exists(trans_path):
         print(f"Error: {trans_path} not found.")
         return
         
    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    print(f"Loaded {len(all_data)} transitions.")

    # Load Latents
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

    # Look for cached latents map
    vae_basename = os.path.basename(VAE_PATH).replace('.pth', '')
    cache_path = os.path.join(DATA_ROOT, f"{vae_basename}_global_latents.pt")
    print(f"Targeting latent cache: {os.path.basename(cache_path)}")
    
    latent_dict = {}
    if os.path.exists(cache_path):
        print("-> Cache found! Loading latents into memory...")
        latent_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("path_map", {})
        print(f"-> Pre-loaded {len(latent_dict)} hashed paths from cache.")
    else:
        print("-> No existing latent cache found. On-the-fly extraction will be used.")
        
    dataset = IRDataset(all_data, latent_dict)
    
    # Extract missing latents for valid dataset items
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    missing_count = 0
    print("Checking dataset geometry and parsing distance labels...")
    for curr, nxt in dataset.transitions:
        p = curr.get('image_path', '')
        if p and p not in latent_dict:
            img_path = os.path.join(DATA_ROOT, p)
            if os.path.exists(img_path):
                missing_count += 1
                if missing_count % 500 == 0:
                    print(f"   [Latent Cache Miss] On-the-fly computing latent for frame: {p}...")
                img = Image.open(img_path).convert('RGB')
                t_img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, mu, _ = vae(t_img)
                latent_dict[p] = mu.cpu().squeeze()
            else:
                pass
                
    if missing_count > 0:
        print(f"Computed {missing_count} total missing latents natively.")

    print(f"\nFinal dataset compiled: {len(dataset)} sequence pairs. Initializing dataloader.")

    # Train Policy
    model = IRPredictorNetwork(state_dim=latent_dim*3, action_dim=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\n--- Training Initialization ---")
    print(f"Model Structure: [{latent_dim*3}-Dim State + 8-Dim Embedded Action] -> MLP -> [1-Dim IR Distance Value]")
    print(f"Batch Size: {batch_size} | Learning Rate: {lr} | Epochs: {epochs}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for latents, actions, next_ir in dataloader:
            latents, actions, next_ir = latents.to(device), actions.to(device), next_ir.to(device)
            
            pred_ir = model(latents, actions)
            loss = criterion(pred_ir, next_ir)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch:02d} | MSE Loss: {total_loss/len(dataloader):.4f}")

    out_path = os.path.join(DATA_ROOT, f"{vae_basename}-hello_world_ir_reflex.pth")
    torch.save(model.state_dict(), out_path)
    print(f"\nIR Reflex Model compiled successfully! Saved to {out_path}")


if __name__ == "__main__":
    train(data_root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
