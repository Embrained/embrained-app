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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.spatial_model import TinyVAE
from backend.training.evaluate_oracles import (
    TemporalNetwork, ForwardNetwork, RNNForwardNetwork, 
    algorithmic_math, neural_logic
)

def run_analysis():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initiating Deviation Analysis...")

    # Load Telemetry
    print("Loading Telemetry Data...")
    telemetry_dict = {}
    master_path = os.path.join(data_root, "master_telemetry.csv")
    if not os.path.exists(master_path):
        master_path = os.path.join(data_root, '..', "master_telemetry.csv")
    if not os.path.exists(master_path):
        print(f"Cannot find {master_path}")
        return
        
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
    print("Loading Trajectories...")
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
            
    # Prepare Models
    import glob
    vae_candidates = glob.glob(os.path.join(data_root, '*-vae_*.pth'))
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    target_vae = vae_candidates[0] if vae_candidates else None
    
    f_candidates = glob.glob(os.path.join(data_root, 'topological_forward_*.pth'))
    target_f_net = None
    if f_candidates:
        f_candidates.sort(key=os.path.getmtime, reverse=True)
        target_f_net = f_candidates[0]
    
    if not target_vae or not target_f_net:
        print("Models missing.")
        return
        
    target_t_net = target_f_net.replace('forward', 'temporal')
    if not os.path.exists(target_t_net):
        target_t_net = target_f_net.replace('.pth', '_temporal.pth')

    print(f"Loading Models...")
    vae_state = torch.load(target_vae, map_location=device, weights_only=True)
    latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(vae_state)
    
    f_state = torch.load(target_f_net, map_location=device, weights_only=True)
    is_rnn = any('rnn.' in k for k in f_state.keys())
    if is_rnn:
        f_net = RNNForwardNetwork(latent_dim).to(device)
    else:
        f_net = ForwardNetwork(latent_dim).to(device)
    f_net.load_state_dict(f_state)
    f_net.eval()
    
    t_net = TemporalNetwork(latent_dim).to(device)
    if os.path.exists(target_t_net):
         t_net.load_state_dict(torch.load(target_t_net, map_location=device, weights_only=True))
    t_net.eval()

    # Load Cache
    vae_basename = os.path.basename(target_vae).replace('.pth', '')
    cache_path = os.path.join(data_root, f"{vae_basename}_global_latents.pt")
    if os.path.exists(cache_path):
        latent_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("ts_map", {})
        cached_latents = torch.stack(list(latent_dict.values())).to(device)
    else:
        print("No latents cache.")
        return

    # Phase 1: Statistical Deviation Analysis
    print("Running Deviation Tests...")
    TEST_COUNT = 3000
    MAX_HORIZON = 40
    
    results = []
    
    with torch.no_grad():
        for _ in range(TEST_COUNT):
            traj = random.choice(trajectories)
            idx1 = random.randint(0, len(traj) - 2)
            max_b = min(len(traj) - 1, idx1 + MAX_HORIZON)
            idx2 = random.randint(idx1 + 1, max_b)
            
            n1 = traj[idx1]
            n2 = traj[idx2]
            n1_prev = traj[idx1 - 1] if idx1 > 0 else n1
            
            ts1_prev = n1_prev['lookup_ts']
            ts1 = n1['lookup_ts']
            ts2 = n2['lookup_ts']
            
            if ts1 not in latent_dict or ts2 not in latent_dict or ts1_prev not in latent_dict:
                continue
                
            t_cur, t_goal = telemetry_dict[ts1], telemetry_dict[ts2]
            l_cur, l_goal = latent_dict[ts1], latent_dict[ts2]
            l_prev = latent_dict[ts1_prev] if is_rnn else None
            
            a_act = algorithmic_math(t_cur, t_goal)
            n_act = neural_logic(f_net, t_net, l_cur, l_goal, device, threshold=3.0, l_prev=l_prev, is_rnn=is_rnn, cached_latents=cached_latents)
            
            match = (a_act == n_act)
            dist = math.hypot(t_goal[0] - t_cur[0], t_goal[1] - t_cur[1])
            
            yaw_cur = math.atan2(t_cur[3], t_cur[2])
            yaw_goal = math.atan2(t_goal[3], t_goal[2])
            head_diff = abs((yaw_goal - yaw_cur + math.pi) % (2 * math.pi) - math.pi)
            
            true_steps = idx2 - idx1
            
            results.append({
                'match': int(match),
                'distance': dist,
                'heading_diff': math.degrees(head_diff),
                'steps': true_steps
            })
            
    df_res = pd.DataFrame(results)
    
    # Visualizations
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. By Distance
    df_res['dist_bin'] = pd.cut(df_res['distance'], bins=5)
    dist_means = df_res.groupby('dist_bin', observed=False)['match'].mean()
    axs[0].bar(dist_means.index.astype(str), dist_means.values, color='skyblue')
    axs[0].set_title('Neural Acc vs Distance')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_ylabel('Match Rate')
    
    # 2. By Heading
    df_res['head_bin'] = pd.cut(df_res['heading_diff'], bins=[0, 30, 90, 150, 180])
    head_means = df_res.groupby('head_bin', observed=False)['match'].mean()
    axs[1].bar(head_means.index.astype(str), head_means.values, color='lightgreen')
    axs[1].set_title('Neural Acc vs Heading Difference (deg)')
    axs[1].set_ylabel('Match Rate')
    
    # 3. By Steps
    df_res['step_bin'] = pd.cut(df_res['steps'], bins=[0, 5, 10, 20, 40])
    step_means = df_res.groupby('step_bin', observed=False)['match'].mean()
    axs[2].bar(step_means.index.astype(str), step_means.values, color='salmon')
    axs[2].set_title('Neural Acc vs Sequence Length (Steps)')
    axs[2].set_ylabel('Match Rate')
    
    plt.tight_layout()
    stats_path = os.path.join(data_root, 'analysis_deviation_stats.png')
    plt.savefig(stats_path)
    plt.close()

    # Phase 2: Latent Trajectory Comparison
    print("Generating Latent Space Comparison...")
    samp_traj = None
    for t in trajectories:
        if len(t) > 25:
            samp_traj = t
            break
            
    if not samp_traj: return
    
    true_latents = []
    pred_latents = []
    
    ts_list = [n['lookup_ts'] for n in samp_traj if n['lookup_ts'] in latent_dict]
    if len(ts_list) < 20: return
    ts_list = ts_list[:20]
    
    with torch.no_grad():
        for i, ts in enumerate(ts_list):
            true_latents.append(latent_dict[ts].cpu().numpy().squeeze())
            
            if i == 0:
                l_cur = latent_dict[ts].unsqueeze(0).to(device)
                l_prev = l_cur
                pred_latents.append(l_cur.cpu().numpy().squeeze())
            else:
                act = 1 # Assume forward for trajectory projection
                if hasattr(samp_traj[i], 'action'):
                     # if available
                     act = samp_traj[i]['action']
                     
                act_t = torch.tensor([act], dtype=torch.long, device=device)
                
                if is_rnn:
                    l_in = torch.stack([l_prev, l_cur], dim=1)
                    l_next = f_net(l_in, act_t)
                    l_prev = l_cur
                    l_cur = l_next
                else:
                    l_next = f_net(l_cur, act_t)
                    l_cur = l_next
                    
                pred_latents.append(l_cur.cpu().numpy().squeeze())

    true_latents = np.array(true_latents)
    pred_latents = np.array(pred_latents)
    
    pca = PCA(n_components=2)
    pca.fit(true_latents)
    
    true_2d = pca.transform(true_latents)
    pred_2d = pca.transform(pred_latents)
    
    plt.figure(figsize=(8, 6))
    plt.plot(true_2d[:, 0], true_2d[:, 1], 'o-', label='True Latent Path')
    plt.plot(pred_2d[:, 0], pred_2d[:, 1], 'x--', label='Neural Forward Pred')
    plt.title('Latent Space Trajectory: True vs Forward Model Prediction')
    plt.legend()
    
    latent_path = os.path.join(data_root, 'analysis_latent_traj.png')
    plt.savefig(latent_path)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    run_analysis()
