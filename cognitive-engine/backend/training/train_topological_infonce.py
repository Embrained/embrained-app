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
import math
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Ensure backend imports work
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR, ACTION_PWM_MAP
from modules.spatial_model import TinyVAE

# Maximum look-ahead distance for Inverse Dynamics
MAX_HORIZON = 20 

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

import torch.nn.functional as F

# --- MODELS ---
class TemporalNetwork(nn.Module):
    """Predicts literal temporal frames (distance) between two latents."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Continuous float representing "steps away"
        )
        
    def forward(self, l1, l2):
        # Cross-Attention: Current State (q) attends to Goal State (k, v)
        q = l1.unsqueeze(1)
        k = l2.unsqueeze(1)
        v = l2.unsqueeze(1)
        attn_out, _ = self.mha(q, k, v)
        attn_out = attn_out.squeeze(1)
        
        x = torch.cat([l1, attn_out], dim=-1)
        return self.net(x)

class ForwardNetwork(nn.Module):
    """Predicts the next future latent state given the current state and action."""
    def __init__(self, latent_dim=32, action_dim=3, action_embed_dim=8):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_embed_dim)
        input_dim = latent_dim + action_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim) # Predicts the latent delta
        )
        
    def forward(self, l1, a1):
        a_emb = self.action_embed(a1)
        x = torch.cat([l1, a_emb], dim=-1)
        return l1 + self.net(x) # Residual dynamics

# --- DATASET ---
class TopologicalDataset(Dataset):
    def __init__(self, trajectories, latent_dict, num_samples):
        self.trajectories = trajectories
        self.latents = latent_dict
        self.num_samples = num_samples
        self.valid_trajs = [t for t in trajectories if len(t) > 5]

    def _get_action(self, node):
        best_action = -1
        if 'macro_action' in node:
            best_action = int(node['macro_action'])
        else:
            raw_l = float(node.get('left_cmd', 0.0))
            raw_r = float(node.get('right_cmd', 0.0))
            best_dist = float('inf')
            for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
                dist = math.hypot(raw_l - map_l, raw_r - map_r)
                if dist < best_dist:
                    best_dist = dist
                    best_action = act_id
        
        # Dubins enforcement mapping (Forward: 1, Left: 3, Right: 4)
        if best_action == 1: return 0
        elif best_action == 3: return 1
        elif best_action == 4: return 2
        else: return -1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        action_A = -1
        while action_A == -1:
            # 1. Random Trajectory
            traj = random.choice(self.valid_trajs)
            t_len = len(traj)
            
            # 2. Pick node A
            idx_A = random.randint(0, t_len - 2)
            node_A = traj[idx_A]
            # Action at Node A answers: "What movement did I initiate here?"
            action_A = self._get_action(node_A)
            
        path_a = node_A.get('image_path', '')
        latent_A = self.latents[path_a]
        
        node_A_Next = traj[idx_A + 1]
        path_a_next = node_A_Next.get('image_path', '')
        latent_A_Next = self.latents[path_a_next]
        
        # 3. Pick node B (Goal in the future)
        # Bounded by MAX_HORIZON
        max_b = min(t_len - 1, idx_A + MAX_HORIZON)
        idx_B = random.randint(idx_A + 1, max_b)
        
        node_B = traj[idx_B]
        path_b = node_B.get('image_path', '')
        latent_B = self.latents[path_b]
        
        # 4. Temporal Distance 
        temporal_dist = torch.tensor([float(idx_B - idx_A)], dtype=torch.float)
        
        return latent_A, latent_A_Next, latent_B, temporal_dist, torch.tensor(action_A, dtype=torch.long)


def train(data_root, num_epochs=20, batch_size=128, learning_rate=1e-4, stop_event=None, progress_callback=None, model_filename=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Topological Distillation pipeline spinning up on: {device}")
    
    DATA_ROOT = os.path.abspath(str(data_root))
    
    import glob
    vae_candidates = glob.glob(os.path.join(DATA_ROOT, '*-vae_*.pth'))
    if not vae_candidates:
        print("CRITICAL ERROR: No trained VAE candidates found!")
        return None, None
        
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    VAE_PATH = vae_candidates[0]
    VAE_MODEL_FILENAME = os.path.basename(VAE_PATH)
    
    # 1. Load Trajectories
    print("Loading all_transitions.json...")
    trans_path = os.path.join(DATA_ROOT, "all_transitions.json")
    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    sessions = {}
    for item in all_data:
        s = item['session']
        if s not in sessions: sessions[s] = []
        sessions[s].append(item)
        
    trajectories = []
    for s in sessions:
        traj = sorted(sessions[s], key=lambda x: x['timestamp'])
        if len(traj) > 5:
            trajectories.append(traj)
            
    print(f"Aggregated {len(trajectories)} episodic trajectories.")

    # 2. Load VAE and Precompute Latents
    print(f"Executing Feature Extraction via {VAE_MODEL_FILENAME}...")
    try:
        vae_state = torch.load(VAE_PATH, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find '{VAE_PATH}'! Are you sure it finished training?")
        return None, None
        
    latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(vae_state)
    vae = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=img_dim, in_channels=in_channels).to(device)
    vae.load_state_dict(vae_state)
    vae.eval()
    
    vae_basename = VAE_MODEL_FILENAME.replace('.pth', '')
    cache_path = os.path.join(DATA_ROOT, f"{vae_basename}_global_latents.pt")
    
    if os.path.exists(cache_path):
        print(f"Loading comprehensively extracted topological mapping from global cache...")
        latent_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("path_map", {})
        print(f"Visual Extraction Complete: {len(latent_dict)} embedded frames loaded.")
    else:
        print(f"CRITICAL ERROR: {cache_path} not found! You must finish VAE training to compile global latents.")
        return None, None

    # 3. Model Init
    temporal_net = TemporalNetwork(latent_dim=latent_dim).to(device)
    forward_net = ForwardNetwork(latent_dim=latent_dim).to(device)
    
    
    # We can optimize them concurrently!
    params = list(temporal_net.parameters()) + list(forward_net.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    dataset = TopologicalDataset(trajectories, latent_dict, num_samples=30000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    print("\n--- Initiating Dual-Module Topological Training (Forward Dynamics: InfoNCE) ---")
    best_loss_f = float('inf')
    best_forward_path = None
    
    for epoch in range(1, num_epochs + 1):
        if stop_event and stop_event.is_set():
            print("Training aborted via stop event.")
            break
        temporal_net.train()
        forward_net.train()
        
        total_t_loss = 0
        total_f_loss = 0
        
        for lA, lA_next, lB, true_dist, true_act in dataloader:
            lA, lA_next, lB = lA.to(device), lA_next.to(device), lB.to(device)
            true_dist = true_dist.to(device)
            true_act = true_act.to(device)
            
            # Module A: Distance Prediction
            pred_dist = temporal_net(lA, lB)
            loss_t = mse_loss(pred_dist, true_dist)
            
            # Module B: Forward Dynamics Prediction (InfoNCE Contrastive Loss)
            pred_lA_next = forward_net(lA, true_act)
            
            temperature = 0.1
            # Normalize vectors
            pred_norm = F.normalize(pred_lA_next, dim=-1)
            true_norm = F.normalize(lA_next, dim=-1)
            
            # Compute similarity logits
            logits = torch.matmul(pred_norm, true_norm.T) / temperature
            # Diagonal holds true positives
            labels = torch.arange(logits.size(0)).to(device)
            
            loss_f = ce_loss(logits, labels)
            
            # Combined Loss
            loss = loss_t + loss_f
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_t_loss += loss_t.item()
            total_f_loss += loss_f.item()
            
        scheduler.step()
        
        avg_t = total_t_loss / len(dataloader)
        avg_f = total_f_loss / len(dataloader)
        
        print(f"Epoch {epoch:02d} | Temporal MSE: {avg_t:.2f} | Forward MSE: {avg_f:.4f}")
        
        if avg_f < best_loss_f:
            best_loss_f = avg_f
            if model_filename:
                base_name = model_filename.replace('.pth', '')
                t_name = f"{base_name}_temporal.pth"
                f_name = f"{base_name}.pth"
            else:
                try:
                    ts = VAE_MODEL_FILENAME.split('_')[-2] + '_' + VAE_MODEL_FILENAME.split('_')[-1].replace('.pth', '')
                except Exception:
                    ts = "default_timestamp"
                t_name = f"topological_temporal_{ts}.pth"
                f_name = f"topological_forward_{ts}.pth"
                
            t_path = os.path.join(DATA_ROOT, t_name)
            f_path = os.path.join(DATA_ROOT, f_name)
            
            torch.save(temporal_net.state_dict(), t_path)
            torch.save(forward_net.state_dict(), f_path)
            best_forward_path = f_path
            
        if progress_callback:
            progress_callback(epoch, avg_f)

    print(f"\nTopological SLAM Assembly Complete. Best Forward InfoNCE Loss: {best_loss_f:.4f}")

    # Run Evaluation
    plot_output_path = None
    try:
        from backend.training.evaluate_oracles import run_evaluation
        plot_output_path = run_evaluation(data_root=DATA_ROOT, eval_model_path=best_forward_path)
    except Exception as e:
        print(f"Evaluation hook failed: {e}")
        
    return best_forward_path, plot_output_path

if __name__ == "__main__":
    test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    train(test_root, num_epochs=35)
