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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.spatial_model import TinyVAE

transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

# --- TOPOLOGICAL ARCHITECTURES ---
class TemporalNetwork(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, l1, l2):
        q = l1.unsqueeze(1)
        k = l2.unsqueeze(1)
        v = l2.unsqueeze(1)
        attn_out, _ = self.mha(q, k, v)
        attn_out = attn_out.squeeze(1)
        x = torch.cat([l1, attn_out], dim=-1)
        return self.net(x)

class ForwardNetwork(nn.Module):
    def __init__(self, latent_dim=32, action_dim=5, action_embed_dim=8):
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
            nn.Linear(128, latent_dim)
        )
    def forward(self, l1, a1):
        a_emb = self.action_embed(a1)
        x = torch.cat([l1, a_emb], dim=-1)
        return l1 + self.net(x)

class RNNForwardNetwork(nn.Module):
    def __init__(self, latent_dim=32, action_dim=5, action_embed_dim=8):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_embed_dim)
        self.rnn = nn.GRU(input_size=latent_dim, hidden_size=128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128 + action_embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, l_seq, a1):
        rnn_out, _ = self.rnn(l_seq)
        last_hidden = rnn_out[:, -1, :]
        a_emb = self.action_embed(a1)
        x = torch.cat([last_hidden, a_emb], dim=-1)
        return l_seq[:, -1, :] + self.fc(x)

# --- LOGIC ORACLES ---
def algorithmic_math(t_cur, t_goal):
    """Raw trig logic stripped of loop pacing constraints"""
    cx, cy, cos_yaw, sin_yaw = t_cur[0], t_cur[1], t_cur[2], t_cur[3]
    gx, gy, gcos, gsin = t_goal[0], t_goal[1], t_goal[2], t_goal[3]
    
    yaw = math.atan2(sin_yaw, cos_yaw)
    g_yaw = math.atan2(gsin, gcos)
    
    dx = gx - cx
    dy = gy - cy
    e_dist = math.hypot(dx, dy)
    angle_to_goal = math.atan2(dy, dx)
    
    heading_err = (angle_to_goal - yaw + math.pi) % (2 * math.pi) - math.pi
    final_heading_err = (g_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
    
    dist_thresh = 0.10
    angle_thresh = 0.3
    
    if e_dist < dist_thresh:
        if abs(final_heading_err) < angle_thresh: return 5
        elif final_heading_err > 0: return 4
        else: return 3
    else:
        if abs(heading_err) < 0.4: return 1
        elif heading_err > 0: return 4
        else: return 3

def neural_logic(forward_net, temporal_net, l_cur, l_goal, device, threshold=1.0, l_prev=None, is_rnn=False, cached_latents=None):
    l_cur = l_cur.unsqueeze(0).to(device)
    l_goal = l_goal.unsqueeze(0).to(device)
    
    mode = 'L2' # 'L2' for far approaches, 'T_NET' for close final alignments
    is_latentslam = hasattr(forward_net, 'predict_next_state')
    
    if temporal_net is not None:
        t_dist_to_final = temporal_net(l_cur, l_goal).item()
        if t_dist_to_final <= 4.0:
            mode = 'T_NET'
            
    if is_rnn:
        if l_prev is None: l_prev = l_cur # Pad if first frame
        else: l_prev = l_prev.unsqueeze(0).to(device)
        l_input_init = torch.stack([l_prev, l_cur], dim=1)
    else:
        l_input_init = l_cur

    with torch.no_grad():
        dist_val = torch.norm(l_cur - l_goal, p=2).item()
        if dist_val < threshold:
            return 5 # Intentional Stop Action
            
        # BFS Trajectory Search (Depth 3)
        queue = []
        for action in [1, 3, 4]:
            if is_latentslam:
                a_idx = 0 if action == 1 else (1 if action == 3 else 2)
                pred_next_l = forward_net.predict_next_state(l_input_init)[0, a_idx].unsqueeze(0)
            else:
                detected_action_dim = forward_net.action_embed.weight.shape[0] if hasattr(forward_net, 'action_embed') else 5
                if detected_action_dim == 5:
                    act_tensor = torch.tensor([action], dtype=torch.long, device=device)
                else:
                    a_idx = 0 if action == 1 else (1 if action == 3 else 2)
                    act_tensor = torch.tensor([a_idx], dtype=torch.long, device=device)
                pred_next_l = forward_net(l_input_init, act_tensor)
            
            if is_rnn:
                new_input = torch.stack([l_input_init[:, -1, :], pred_next_l], dim=1)
            else:
                new_input = pred_next_l
                
            queue.append((new_input, pred_next_l, action, 1))

        best_action = 0
        min_dist = float('inf')
        
        while queue:
            l_input, raw_l, first_act, depth = queue.pop(0)
            
            if depth == 3: # Horizon Reached
                if mode == 'T_NET':
                    dist_val = temporal_net(raw_l, l_goal).item()
                else:
                    dist_val = torch.norm(raw_l.squeeze() - l_goal.squeeze(), p=2).item()
                    
                if dist_val < min_dist:
                    min_dist = dist_val
                    best_action = first_act
            else:
                for action in [1, 3, 4]:
                    if is_latentslam:
                        a_idx = 0 if action == 1 else (1 if action == 3 else 2)
                        pred_next_l = forward_net.predict_next_state(l_input.unsqueeze(0))[0, a_idx].unsqueeze(0)
                    else:
                        detected_action_dim = forward_net.action_embed.weight.shape[0] if hasattr(forward_net, 'action_embed') else 5
                        if detected_action_dim == 5:
                            act_tensor = torch.tensor([action], dtype=torch.long, device=device)
                        else:
                            a_idx = 0 if action == 1 else (1 if action == 3 else 2)
                            act_tensor = torch.tensor([a_idx], dtype=torch.long, device=device)
                        pred_next_l = forward_net(l_input, act_tensor)
                    
                    if is_rnn:
                        new_input = torch.stack([l_input[:, -1, :], pred_next_l], dim=1)
                    else:
                        new_input = pred_next_l
                        
                    queue.append((new_input, pred_next_l, first_act, depth + 1))
                    
        return best_action

def run_evaluation(data_root=None, eval_model_path=None):
    if data_root is None:
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initiating Pipeline Evaluation...")

    # Load Telemetry
    print("Loading Golden Telemetry Data...")
    telemetry_dict = {}
    master_path = os.path.join(data_root, "master_telemetry.csv")
    if not os.path.exists(master_path):
        master_path = os.path.join(data_root, '..', "master_telemetry.csv")
        
    df = pd.read_csv(master_path)
    for _, row in df.iterrows():
        yaw_rad = math.radians(row['yaw_deg'])
        telemetry_dict[str(row['ts'])] = [
            row['cx'] / 640.0,
            row['cy'] / 480.0,
            math.cos(yaw_rad),
            math.sin(yaw_rad)
        ]

    # Load Datasets
    print("Loading Validation Trajectories...")
    trans_path = os.path.join(data_root, "all_transitions.json")
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
        valid_nodes = []
        for n in traj:
            img_path = n.get('image_path', '')
            ts = os.path.basename(img_path).replace('frame_', '').replace('.jpg', '')
            if ts in telemetry_dict:
                n['lookup_ts'] = ts
                valid_nodes.append(n)
        if len(valid_nodes) > 5:
            trajectories.append(valid_nodes)
            
    print(f"Aggregated {len(trajectories)} matching episodic trajectories.")

    # Prepare Dynamic Loader
    import glob
    vae_candidates = glob.glob(os.path.join(data_root, '*-vae_*.pth'))
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    target_vae = vae_candidates[0] if vae_candidates else None
    
    target_f_net = eval_model_path
    if not target_f_net:
        f_candidates = glob.glob(os.path.join(data_root, 'topological_forward_*.pth'))
        if f_candidates:
            f_candidates.sort(key=os.path.getmtime, reverse=True)
            target_f_net = f_candidates[0]
    
    if not target_vae or not target_f_net:
        print("Required models missing for evaluation. Aborting.")
        return None
        
    target_t_net = target_f_net.replace('forward', 'temporal')
    if not os.path.exists(target_t_net):
        target_t_net = target_f_net.replace('.pth', '_temporal.pth')

    print(f"Mounting Neural Models from explicitly bound pair...")
    
    # Safely load the matching VAE
    vae_state = None
    for cand in vae_candidates:
        try:
            tmp_state = torch.load(cand, map_location=device, weights_only=True)
            latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(tmp_state)
            vae_state = tmp_state
            target_vae = cand
            break
        except KeyError:
            continue
            
    if vae_state is None:
        print("Failed to find compatible continuous VAE for evaluation!")
        return None
        
    vae = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=img_dim, in_channels=in_channels).to(device)
    vae.load_state_dict(vae_state)
    vae.eval()
    
    t_net = TemporalNetwork(latent_dim).to(device)
    if os.path.exists(target_t_net):
         t_net.load_state_dict(torch.load(target_t_net, map_location=device, weights_only=True))
    t_net.eval()
    
    f_state = torch.load(target_f_net, map_location=device, weights_only=True)
    is_rnn = any('rnn.' in k for k in f_state.keys())
    is_latentslam = any('transition_model.' in k for k in f_state.keys())
    
    
    detected_action_dim = 5
    if 'action_embed.weight' in f_state:
        detected_action_dim = f_state['action_embed.weight'].shape[0]

    if is_latentslam:
        print("Detected LatentSLAM Architecture.")
        hidden_dim = f_state['transition_model.0.weight'].shape[0] if 'transition_model.0.weight' in f_state else 256
        out_features = f_state['transition_model.4.weight'].shape[0] if 'transition_model.4.weight' in f_state else latent_dim * 3
        num_acts = out_features // latent_dim
        
        if 'fc_e.weight' in f_state:
            from backend.models.quantized_spatial import DiscreteLatentSLAM
            f_net = DiscreteLatentSLAM(latent_dim=latent_dim, num_actions=num_acts, hidden_dim=hidden_dim, model_size=model_size, image_size=img_dim).to(device)
        else:
            from backend.models.latentslam import LatentSLAM
            f_net = LatentSLAM(latent_dim=latent_dim, num_actions=num_acts, hidden_dim=hidden_dim, model_size=model_size, image_size=img_dim).to(device)
            
        f_net.load_state_dict(f_state, strict=False)
    elif is_rnn:
        print("Detected Temporal RNN Architecture. Loading Sequence Memory Unit...")
        f_net = RNNForwardNetwork(latent_dim=latent_dim, action_dim=detected_action_dim).to(device)
        f_net.load_state_dict(f_state)
    else:
        print("Detected Spatial-Only ML Architecture.")
        f_net = ForwardNetwork(latent_dim=latent_dim, action_dim=detected_action_dim).to(device)
        f_net.load_state_dict(f_state)
    f_net.eval()

    # Pre-encode all frames 
    vae_basename = os.path.basename(target_vae).replace('.pth', '')
    cache_path = os.path.join(data_root, f"{vae_basename}_global_latents.pt")
    
    if os.path.exists(cache_path):
        print(f"Loading comprehensively extracted topological mapping from global cache...")
        latent_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("ts_map", {})
        print(f"Loaded {len(latent_dict)} latent evaluations.")
        cached_latents = torch.stack(list(latent_dict.values())).to(device)
    else:
        print(f"CRITICAL ERROR: {cache_path} not found! You must finish VAE training to compile global latents.")
        return

    # Batch Process Evaluation
    print("Running dual inference collision loop...")
    
    # Run 1,500 contextual evaluations across optimal sequence lengths
    TEST_COUNT = 1500
    MAX_HORIZON = 2
    
    # 1. Pre-generate deterministic test batch from explicit short trajectories
    test_batch = []
    while len(test_batch) < TEST_COUNT:
        traj = random.choice(trajectories)
        if len(traj) < 3: continue
        
        idx1 = random.randint(0, len(traj) - 2)
        step_sz = random.randint(1, min(MAX_HORIZON, len(traj) - 1 - idx1))
        idx2 = idx1 + step_sz
        
        # Ground truth is explicit action taken at idx1
        truth_act = traj[idx1].get('macro_action', 0)
        if truth_act not in [1, 3, 4]: continue
        
        test_batch.append((traj, idx1, idx2, truth_act))
        
    print("Disabling Neural Targeting Threshold to force spatial prediction...")
    threshold_l2 = -1.0
    
    temporal_distances = []
    valid_batch = []
    
    with torch.no_grad():
        for traj, idx1, idx2, truth_act in test_batch:
            n1 = traj[idx1]
            n2 = traj[idx2]
            
            n1_prev = traj[idx1 - 1] if idx1 > 0 else n1
            
            ts1_prev = n1_prev['lookup_ts']
            ts1 = n1['lookup_ts']
            ts2 = n2['lookup_ts']
            
            if ts1 not in latent_dict or ts2 not in latent_dict or ts1_prev not in latent_dict:
                continue
                
            l_cur, l_goal = latent_dict[ts1], latent_dict[ts2]
                
            dist_val = torch.norm(l_cur - l_goal, p=2).item()
            temporal_distances.append(dist_val)
            valid_batch.append((truth_act, l_cur, l_goal, latent_dict[ts1_prev] if is_rnn else None))
            
    temporal_distances.sort()
    print(f"Sampled optimal paths. Forced Spatial Output Priority active.")
    
    confusion_matrix = np.zeros((6, 6), dtype=int)

    neural_stops = 0
    for i, (truth_act, l_cur, l_goal, l_prev) in enumerate(valid_batch):
        if i % 100 == 0:
            print(f"Evaluating Path {i}/{len(valid_batch)}...", end="\r")
            
        # Determine Ground Truth Route vs Neural Route
        n_act = neural_logic(f_net, t_net, l_cur, l_goal, device, threshold=threshold_l2, l_prev=l_prev, is_rnn=is_rnn, cached_latents=cached_latents)
        if n_act == 5:
            neural_stops += 1
            
        confusion_matrix[truth_act, n_act] += 1
        
    print(f"\nNeural Oracle emitted {neural_stops} Stops.")

    # Validation Rendering - Confusion Matrix
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('#f8fafc')
    
    # Isolate explicit [Forward, Left, Right] indices to match Neural outputs
    plot_indices = [1, 3, 4]
    plot_matrix = confusion_matrix[np.ix_(plot_indices, plot_indices)]
    
    action_names = ['Forward (1)', 'Left (3)', 'Right (4)']
    
    # Calculate row percentages for annotations
    row_sums = plot_matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(plot_matrix, row_sums, out=np.zeros_like(plot_matrix, dtype=float), where=row_sums!=0)
    
    # Create custom annotations with both count and percentage
    annot_data = np.empty_like(plot_matrix, dtype=object)
    for r in range(3):
        for c in range(3):
            annot_data[r, c] = f"{plot_matrix[r, c]}\n({percentages[r, c]*100:.1f}%)"
            
    ax = sns.heatmap(plot_matrix, annot=annot_data, fmt='', cmap='Blues', 
                xticklabels=action_names, yticklabels=action_names, cbar=False,
                annot_kws={"size": 11, "weight": "bold"})
                
    ax.set_title("Neural Oracle Parity vs Optimal Autonomous Control", fontsize=15, fontweight='bold', pad=20, color='#1e293b')
    ax.set_ylabel("True Physics Trajectory (Ground Truth)", fontsize=12, labelpad=10, color='#475569', fontweight='bold')
    ax.set_xlabel("Neural Latent Space (Predicted Execution)", fontsize=12, labelpad=10, color='#475569', fontweight='bold')
    
    plt.tight_layout()
    
    basename = target_f_net.replace('.pth', '')
    if '\\' in basename: basename = basename.split('\\')[-1]
    if '/' in basename: basename = basename.split('/')[-1]
    
    arti_path = os.path.join(data_root, f"{basename}_parity.png")
    plt.savefig(arti_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nParity Analysis chart exported natively to: {arti_path}")
    return arti_path

if __name__ == "__main__":
    run_evaluation()
