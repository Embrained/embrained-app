import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_DIR, MODELS_DIR
from modules.spatial_model import DiscreteVQVAE

class TemporalTripletDataset(Dataset):
    def __init__(self, trajectories, data_root, transform):
        self.samples = []
        self.data_root = data_root
        self.transform = transform
        
        # Build samples: Anchor (t), Positive (t+1), Negative (random)
        for traj in trajectories:
            for i in range(len(traj) - 1):
                anchor = traj[i]
                positive = traj[i+1]
                self.samples.append((anchor, positive))
                
        # To sample negatives easily, flatten all nodes
        self.all_nodes = [node for traj in trajectories for node in traj]
                
    def _load_img(self, node):
        try:
            if 'image_path' in node:
                p = node['image_path']
                if not os.path.isabs(p):
                    p = os.path.join(self.data_root, p)
                
                # [FIX] Explicitly reject static external webcam frames to prevent degenerate codebook collapse
                if 'webcam_frame_' in p:
                    raise ValueError("Rejecting webcam frame")
                if 'frame_' not in os.path.basename(p):
                    raise ValueError("Not a valid onboard frame")

                if os.path.exists(p):
                    img = cv2.imread(p)
                    if img is not None:
                        return self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
        return torch.zeros((3, 64, 64))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_node, pos_node = self.samples[idx]
        # Random negative from all nodes
        neg_node = random.choice(self.all_nodes)
        
        anchor_img = self._load_img(anchor_node)
        pos_img = self._load_img(pos_node)
        neg_img = self._load_img(neg_node)
        
        return anchor_img, pos_img, neg_img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting VQ-VAE + InfoNCE Contrastive Training on {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    trans_path = os.path.join(DATA_ROOT, "all_transitions.json")
    
    if not os.path.exists(trans_path):
        print(f"Error: {trans_path} not found.")
        return
        
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
            
    print(f"Loaded {len(trajectories)} valid trajectories.")
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.ToTensor(),
    ])
    
    dataset = TemporalTripletDataset(trajectories, DATA_ROOT, transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    model = DiscreteVQVAE(latent_dim=32, model_size='large', input_spatial_dim=64, num_embeddings=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # InfoNCE Temperature
    temperature = 0.1
    mse_weight = 1.0
    contrastive_weight = 1.0
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_vq = 0
        total_mse = 0
        total_infonce = 0
        
        for batch_idx, (anchor, pos, neg) in enumerate(dataloader):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass anchor (to get reconstruction and vq loss)
            recon, q_anchor, vq_loss_a, _ = model(anchor)
            _, q_pos, vq_loss_p, _ = model(pos)
            _, q_neg, vq_loss_n, _ = model(neg)
            
            vq_loss = (vq_loss_a + vq_loss_p + vq_loss_n) / 3.0
            mse_loss = F.mse_loss(recon, anchor)
            
            # InfoNCE Loss
            # Cosine similarity
            q_anchor_norm = F.normalize(q_anchor, dim=-1)
            q_pos_norm = F.normalize(q_pos, dim=-1)
            q_neg_norm = F.normalize(q_neg, dim=-1)
            
            pos_sim = torch.sum(q_anchor_norm * q_pos_norm, dim=-1) / temperature
            neg_sim = torch.sum(q_anchor_norm * q_neg_norm, dim=-1) / temperature
            
            # Contrastive loss: -log( exp(pos) / (exp(pos) + exp(neg)) )
            # Numerically stable:
            logits = torch.stack([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            infonce_loss = F.cross_entropy(logits, labels)
            
            loss = vq_loss + (mse_weight * mse_loss) + (contrastive_weight * infonce_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_vq += vq_loss.item()
            total_mse += mse_loss.item()
            total_infonce += infonce_loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(dataloader):.4f} | InfoNCE: {total_infonce/len(dataloader):.4f} | MSE: {total_mse/len(dataloader):.4f} | VQ: {total_vq/len(dataloader):.4f}")
        
    out_path = os.path.join(MODELS_DIR, "vqvae_infonce.pth")
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved trained DiscreteVQVAE to {out_path}")

if __name__ == "__main__":
    main()
