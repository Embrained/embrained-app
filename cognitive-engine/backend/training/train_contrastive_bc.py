import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.training.datasets.latentslam_dataset import LatentSLAMDataset
from modules.spatial_model import FixedGoalBCNetwork, TinyVAE

def infonce_loss(features_q, features_k, temperature=0.1):
    """SimCLR-style InfoNCE matching positive pairs and repelling cross-batch negatives."""
    q = F.normalize(features_q, dim=1)
    k = F.normalize(features_k, dim=1)
    logits = torch.matmul(q, k.T) / temperature # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def normalize_path(p):
    return p.replace('\\', '/')

class EndToEndContrastiveBC(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4):
        super(EndToEndContrastiveBC, self).__init__()
        # Explicit TinyVAE instantiation so dictionary keys map perfectly for engine.py hot-swap
        self.vision = TinyVAE(latent_dim=latent_dim, model_size='large')
        
        # Identical core to standalone FixedGoalBCNetwork
        self.policy = FixedGoalBCNetwork(state_dim=latent_dim, action_dim=action_dim)
        
    def forward_features(self, x):
        _, mu, _ = self.vision(x)
        return mu
        
    def forward(self, x):
        z = self.forward_features(x)
        return self.policy(z)

def train(epochs=50, batch_size=32, lr=1e-4, contrastive_weight=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting End-to-End Contrastive BC Training on {device}")
    
    DATA_ROOT = "data"
    GOALS_DIR = os.path.join(DATA_ROOT, "goals")
    DIST_THRESHOLD = 1.50 # Hard BC Threshold
    
    # --- 1. Dataset Filtering via Geometric VAE ---
    # We still use the frozen VAE geometry just to logically filter trajectories!
    vae_files = [f for f in os.listdir(DATA_ROOT) if f.startswith('tinyvae') and f.endswith('.pth')]
    vae_files = [f for f in vae_files if not any(x in f.lower() for x in ['hello_world', 'cql', 'reflex', 'fixed_goal', 'policy'])]
    if not vae_files: return print("No VAE found for trajectory filtering.")
    latest_vae = sorted(vae_files, key=lambda f: os.path.getmtime(os.path.join(DATA_ROOT, f)))[-1]
    VAE_PATH = os.path.join(DATA_ROOT, latest_vae)
    
    vae = TinyVAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device, weights_only=True))
    vae.eval()
    
    target_normalized = [
        normalize_path(p) for p in [
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
    ]
    
    # Fast latent extraction for goals
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    target_latents = []
    with torch.no_grad():
        for target in target_normalized:
            p = os.path.join(DATA_ROOT, target)
            if os.path.exists(p):
                img = Image.open(p).convert('RGB')
                t_img = transform(img).unsqueeze(0).to(device)
                _, mu, _ = vae(t_img)
                target_latents.append(mu.cpu().squeeze())
    if not target_latents: return print("No latents recovered.")
    target_latents = torch.stack(target_latents).to(device)
    centroid = target_latents.mean(dim=0)
    
    # Isolate expert trajectories targeting the bounds
    trans_path = os.path.join(DATA_ROOT, "all_transitions.json")
    with open(trans_path, 'r') as f:
        transitions = json.load(f)
        
    sessions = {}
    for t in transitions:
        s = t.get('session', 'default')
        sessions.setdefault(s, []).append(t)
        
    T_HORIZON = 5
    expert_transitions = []
    
    # Using existing latents logic simplified for speed
    cache_path = os.path.join(DATA_ROOT, f"{latest_vae.replace('.pth', '')}_global_latents.pt")
    latent_dict = {}
    if os.path.exists(cache_path):
        raw = torch.load(cache_path, map_location='cpu', weights_only=True).get("path_map", {})
        latent_dict = {normalize_path(k): v.to(device) for k, v in raw.items()}
        
    for s_name, traj in sessions.items():
        traj = sorted(traj, key=lambda x: x['timestamp'])
        
        target_indices = []
        for i, node in enumerate(traj):
            p = normalize_path(node.get('image_path', ''))
            z = latent_dict.get(p)
            if z is not None:
                dists = torch.norm(target_latents - z.unsqueeze(0), dim=1)
                min_dist = torch.min(dists).item()
                if min_dist <= DIST_THRESHOLD:
                    target_indices.append(i)
                    
        for t_idx in target_indices:
            start_idx = max(0, t_idx - T_HORIZON)
            for j in range(start_idx, t_idx):
                act = traj[j].get('macro_action', -1)
                if act in [1, 2, 3, 4]:
                    expert_transitions.append(traj[j])
                    
    expert_nodes = list({id(n): n for n in expert_transitions}.values())
    
    print(f"Subsampled {len(expert_nodes)} expert transition steps hitting exactly the goal bound.")
    if len(expert_nodes) == 0: return

    # --- 2. Build InfoNCE + BC End-to-End Pipeline ---
    dataset = LatentSLAMDataset(expert_nodes, data_root=DATA_ROOT, device=device, image_size=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    model = EndToEndContrastiveBC(latent_dim=32, action_dim=4).to(device)
    model.vision.load_state_dict(vae.state_dict()) # Bootstrap with pixel-fidelity geometry
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n--- Initiating Dual-Module E2E Contrastive Training (InfoNCE + CE) ---")
    
    for ep in range(epochs):
        model.train()
        total_loss = 0
        bc_loss_tot = 0
        nce_loss_tot = 0
        correct = 0
        tot = 0
        
        for img_t, action_t, img_next in loader:
            img_t = img_t.to(device)
            img_next = img_next.to(device)
            action_t = action_t.to(device)
            
            optimizer.zero_grad()
            
            # Extract features computationally
            z_curr = model.forward_features(img_t)
            z_next = model.forward_features(img_next)
            
            # Loss 1: InfoNCE (Positives = Sequential steps, Negatives = Rest of batch)
            loss_infonce = infonce_loss(z_curr, z_next.detach(), temperature=0.1)
            
            # Loss 2: Behavioral Cross Entropy (Padding z_curr x3 to match the fixed goal network's 96-dim expectation)
            z_curr_stacked = torch.cat([z_curr, z_curr, z_curr], dim=-1)
            act_pred = model.policy(z_curr_stacked)
            loss_bc = F.cross_entropy(act_pred, action_t)
            
            loss = loss_bc + contrastive_weight * loss_infonce
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            bc_loss_tot += loss_bc.item()
            nce_loss_tot += loss_infonce.item()
            
            preds = torch.argmax(act_pred, dim=-1)
            correct += (preds == action_t).sum().item()
            tot += action_t.size(0)
            
        acc = correct / max(tot, 1)
        print(f"Epoch {ep+1:02d} | Loss: {total_loss/len(loader):.4f} (BC: {bc_loss_tot/len(loader):.4f}, NCE: {nce_loss_tot/len(loader):.4f}) | Acc: {acc*100:.1f}%")
        
    print("Training Complete! Formatting exact DrQ hybrid checkpoint topology...")
    
    # 3. Formulate the Hot-Swappable Dictionary
    out_dict = {
        'encoder_state_dict': model.vision.state_dict(),
        'model_state_dict': model.policy.state_dict()
    }
    
    out_filename = latest_vae.replace('.pth', '-e2e_contrastive_fixed_goal_bc_model.pth')
    out_path = os.path.join(DATA_ROOT, out_filename)
    torch.save(out_dict, out_path)
    
    print(f"Successfully baked completely hot-swappable artifact: {out_filename}")

if __name__ == '__main__':
    train()
