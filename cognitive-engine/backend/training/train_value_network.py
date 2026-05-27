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
import networkx as nx
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_DIR, MODELS_DIR
from backend.training.train_fixed_goal import TARGET_IMAGES, normalize_path
from modules.spatial_model import TinyVAE, ValueNetwork

class ValueNetworkDataset(Dataset):
    def __init__(self, data_root, vae_path, device):
        self.data_root = data_root
        self.device = device
        self.samples = []
        
        # Load VAE
        self.vae_state = torch.load(vae_path, map_location=device, weights_only=True)
        self.latent_dim, self.model_size, self.img_dim, self.in_channels = TinyVAE.detect_size(self.vae_state)
        self.vae = TinyVAE(latent_dim=self.latent_dim, model_size=self.model_size, input_spatial_dim=self.img_dim, in_channels=self.in_channels).to(device)
        self.vae.load_state_dict(self.vae_state)
        self.vae.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_dim, self.img_dim)),
            T.ToTensor(),
        ])
        
        self._build_graph_and_targets()
        
    def _load_img(self, node):
        p = node.get('image_path', '')
        if not os.path.isabs(p):
            p = os.path.join(self.data_root, p)
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                return self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return torch.zeros((3, self.img_dim, self.img_dim))

    def _get_latent(self, node):
        img = self._load_img(node).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, mu, _ = self.vae(img)
        return mu.squeeze().cpu()

    def _build_graph_and_targets(self):
        trans_path = os.path.join(self.data_root, "all_transitions.json")
        with open(trans_path, 'r') as f:
            all_data = json.load(f)
            
        sessions = {}
        for item in all_data:
            s = item['session']
            if s not in sessions: sessions[s] = []
            sessions[s].append(item)
            
        # Build networkx directed graph
        G = nx.DiGraph()
        nodes_data = {}
        
        print("Building transition graph...")
        for s, seq in sessions.items():
            seq = sorted(seq, key=lambda x: x['timestamp'])
            for i in range(len(seq) - 1):
                u = seq[i]
                v = seq[i+1]
                u_path = u.get('image_path', '')
                v_path = v.get('image_path', '')
                if not u_path or not v_path: continue
                G.add_edge(u_path, v_path)
                nodes_data[u_path] = u
                nodes_data[v_path] = v

        print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        
        target_normalized = [normalize_path(p) for p in TARGET_IMAGES]
        terminal_nodes = []
        for n in G.nodes():
            for t in target_normalized:
                if t in n:
                    terminal_nodes.append(n)
                    
        if not terminal_nodes:
            print("Warning: No target images found in dataset. Picking a random node as goal.")
            terminal_nodes = [list(G.nodes())[-1]]
            
        print(f"Found {len(terminal_nodes)} terminal goal states.")
        
        R = G.reverse()
        distances = {n: float('inf') for n in G.nodes()}
        for t in terminal_nodes:
            distances[t] = 0
            
        queue = deque(terminal_nodes)
        visited = set(terminal_nodes)
        
        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            
            for neighbor in R.neighbors(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = curr_dist + 1
                    queue.append(neighbor)
                    
        print(f"BFS completed. {len(visited)}/{G.number_of_nodes()} states reachable to goal.")
        
        print("Precomputing latents and targets...")
        target_latents = []
        for t in terminal_nodes:
            target_latents.append(self._get_latent(nodes_data[t]))
        if target_latents:
            self.goal_latent = torch.stack(target_latents).mean(dim=0)
        else:
            self.goal_latent = torch.zeros(self.latent_dim)
            
        for u in G.nodes():
            if distances[u] == float('inf'): continue
            
            self.samples.append({
                'u_node': nodes_data[u],
                'dist': distances[u]
            })
            
        print(f"Prepared {len(self.samples)} Value Network samples for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        u_latent = self._get_latent(sample['u_node'])
        target = torch.tensor([sample['dist']], dtype=torch.float)
        return u_latent, self.goal_latent, target

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Value Network Topological Training on {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    import glob
    vae_candidates = glob.glob(os.path.join(DATA_ROOT, '*-vae_*.pth'))
    vae_candidates = [f for f in vae_candidates if not any(x in f.lower() for x in ['hello_world', 'cql', 'reflex', 'fixed_goal', 'policy'])]
    if not vae_candidates:
        print("Error: No VAE found!")
        return
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    VAE_PATH = vae_candidates[0]
    print(f"Using VAE: {VAE_PATH}")
    
    dataset = ValueNetworkDataset(DATA_ROOT, VAE_PATH, device)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    
    value_net = ValueNetwork(latent_dim=dataset.latent_dim).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=1e-4)
    
    num_epochs = 30
    for epoch in range(num_epochs):
        value_net.train()
        total_loss = 0
        
        for batch_idx, (z_cur, z_goal, target) in enumerate(dataloader):
            z_cur, z_goal, target = z_cur.to(device), z_goal.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred_steps = value_net(z_cur, z_goal)
            
            # MSE loss on predicted steps to goal
            loss = F.mse_loss(pred_steps, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Regression Loss: {total_loss/len(dataloader):.4f}")
        
    out_path = os.path.join(DATA_ROOT, "topological_value_network.pth")
    torch.save(value_net.state_dict(), out_path)
    print(f"✅ Saved trained ValueNetwork to {out_path}")

if __name__ == "__main__":
    main()
