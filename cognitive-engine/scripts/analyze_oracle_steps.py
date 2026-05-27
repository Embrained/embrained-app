import os
import sys
import json
import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.spatial_model import TinyVAE
from backend.training.evaluate_oracles import (
    TemporalNetwork, ForwardNetwork, RNNForwardNetwork, 
    algorithmic_math, neural_logic
)

def run_analysis():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initiating Steps-to-Goal Deviation Analysis...")

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
        try:
            yaw_rad = math.radians(row['yaw_deg'])
            telemetry_dict[str(row['ts'])] = [
                row['cx'] / 640.0,
                row['cy'] / 480.0,
                math.cos(yaw_rad),
                math.sin(yaw_rad)
            ]
        except KeyError:
            continue

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

    # Tracking arrays
    results = []
    MAX_ALGO_LEN = 20
    SAMPLES_TO_DRAW = 2000

    print("Iterating over trajectory segments...")
    pairs_tested = 0

    # Collect all valid pairs randomly to ensure coverage
    valid_pairs = []
    for t_idx, traj in enumerate(trajectories):
        for i in range(len(traj) - 2):
            for k in range(i + 1, min(len(traj), i + 35)):
                ts_start = traj[i]['lookup_ts']
                ts_goal = traj[k]['lookup_ts']
                ts_start_prev = traj[i - 1]['lookup_ts'] if i > 0 else ts_start
                if ts_start in latent_dict and ts_goal in latent_dict and ts_start_prev in latent_dict:
                    valid_pairs.append((traj, i, k, ts_start, ts_goal, ts_start_prev))

    random.shuffle(valid_pairs)

    with torch.no_grad():
        for traj, i, k, ts_start, ts_goal, ts_start_prev in valid_pairs:
            if pairs_tested >= SAMPLES_TO_DRAW: break

            t_start = telemetry_dict[ts_start]
            t_goal = telemetry_dict[ts_goal]
            l_start = latent_dict[ts_start]
            l_goal = latent_dict[ts_goal]
            l_start_prev = latent_dict[ts_start_prev] if is_rnn else None

            # 1. Determine baseline match action at step 0
            algo_act = algorithmic_math(t_start, t_goal)
            n_act = neural_logic(f_net, t_net, l_start, l_goal, device, threshold=3.0, l_prev=l_start_prev, is_rnn=is_rnn, cached_latents=cached_latents)
            first_step_match = (algo_act == n_act)

            # 2. Iterate Algorithmic Math along trajectory to find stop
            algo_len = None
            for j in range(i, len(traj)):
                ts_j = traj[j]['lookup_ts']
                if ts_j not in telemetry_dict: continue
                act_j = algorithmic_math(telemetry_dict[ts_j], t_goal)
                if act_j == 5:
                    algo_len = j - i
                    break
            
            # Skip if we didn't naturally find a stop within parameters
            if algo_len is None or algo_len == 0 or algo_len > MAX_ALGO_LEN:
                continue

            # 3. Iterate Neural Math along trajectory to find stop
            neural_len = None
            for j in range(i, len(traj)):
                ts_j = traj[j]['lookup_ts']
                ts_prev_j = traj[j-1]['lookup_ts'] if j > 0 else ts_j
                
                if ts_j not in latent_dict or ts_prev_j not in latent_dict: continue
                    
                n_act_j = neural_logic(
                    f_net, t_net, 
                    latent_dict[ts_j], l_goal, 
                    device, threshold=3.0, 
                    l_prev=latent_dict[ts_prev_j] if is_rnn else None, 
                    is_rnn=is_rnn, cached_latents=cached_latents
                )
                if n_act_j == 5:
                    neural_len = j - i
                    break

            if neural_len is None:
                 neural_len = MAX_ALGO_LEN + 15 # Tally hit upper ceiling if it never completed

            results.append({
                'algo_len': algo_len,
                'neural_len': neural_len,
                'first_step_match': int(first_step_match)
            })
            pairs_tested += 1
            if pairs_tested % 100 == 0:
                print(f"Processed {pairs_tested} segments...")

    df_res = pd.DataFrame(results)

    # Plot 1: Accuracy (First Step Match Rate) by Algorithmic Length
    plt.figure(figsize=(10, 5))
    acc_by_len = df_res.groupby('algo_len')['first_step_match'].mean()
    sns.barplot(x=acc_by_len.index, y=acc_by_len.values, color='cornflowerblue')
    plt.title('Accuracy (First Step Match) vs Target Horizon')
    plt.xlabel('Algorithmic Oracle Horizon (Steps to Goal)')
    plt.ylabel('Match Rate')
    plt.ylim(0, 1.0)
    acc_path = os.path.join(data_root, 'analysis_steps_accuracy.png')
    plt.savefig(acc_path)
    plt.close()

    # Plot 2: Neural Oracle Steps required vs Algorithmic Steps required
    plt.figure(figsize=(10, 5))
    steps_by_len = df_res.groupby('algo_len')['neural_len'].mean()
    
    # Calculate confidence interval safely
    grouped = df_res.groupby('algo_len')['neural_len']
    means = grouped.mean()
    # Fill std with 0 if only one sample
    stds = grouped.std().fillna(0)
    counts = grouped.count()
    
    # Calculate standard error, handle div by zero
    ses = stds / np.sqrt(counts.replace(0, 1))
    
    plt.plot(means.index, means.values, marker='o', label='Mean Neural Steps', color='tomato')
    plt.fill_between(means.index, means - ses, means + ses, alpha=0.3, color='tomato')
    
    # Add identity line
    plt.plot(means.index, means.index, 'k--', label='Perfect Parity (y=x)')
    
    plt.title('Execution Steps: Neural vs Algorithmic Parity')
    plt.xlabel('Algorithmic Oracle Length (Target Steps)')
    plt.ylabel('Neural Oracle Steps Required')
    plt.legend()
    
    steps_path = os.path.join(data_root, 'analysis_steps_parity.png')
    plt.savefig(steps_path)
    plt.close()

    print("Done!")

if __name__ == "__main__":
    run_analysis()
