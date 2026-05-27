import os
import sys
import json
import glob
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.training.evaluate_oracles import algorithmic_math, neural_logic
from backend.models.latentslam import LatentSLAM
try:
    from backend.models.quantized_spatial import DiscreteLatentSLAM
except ImportError:
    DiscreteLatentSLAM = None

def main():
    data_dir = r"C:\Users\chris\Embrained\software_suite\data"
    output_dir = r"C:\Users\chris\.gemini\antigravity\brain\f38af09f-2bd0-489f-9f34-71894172bea0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Telemetry...")
    telemetry_dict = {}
    master_path = os.path.join(data_dir, "master_telemetry.csv")
    df = pd.read_csv(master_path)
    for _, row in df.iterrows():
        import math
        yaw_rad = math.radians(row['yaw_deg'])
        telemetry_dict[str(row['ts'])] = [
            row['cx'] / 640.0,
            row['cy'] / 480.0,
            math.cos(yaw_rad),
            math.sin(yaw_rad)
        ]

    print("Loading Valid Trajectories...")
    trans_path = os.path.join(data_dir, "all_transitions.json")
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
            
    print("Loading Oracles...")
    fwd_files = glob.glob(os.path.join(data_dir, "topological_forward_latentslam_*.pth"))
    fwd_files.sort(key=os.path.getmtime, reverse=True)
    fwd_files = fwd_files[:2] # Limit to 2 for speed
    
    cache_files = glob.glob(os.path.join(data_dir, "*-vae_*_global_latents.pt"))
    cache_files.sort(key=os.path.getmtime, reverse=True)
    
    if not cache_files:
        print("ERROR: No cache found.")
        return
        
    cache_path = cache_files[0]
    latent_data = torch.load(cache_path, map_location='cpu', weights_only=True)
    global_ts_map = latent_data.get("ts_map", {})
    
    oracles = []
    
    oracles.append({
        'name': "Algorithmic Ground Truth",
        'uid': 'Algorithmic',
        'is_neural': False
    })
    
    for idx, fwd_path in enumerate(fwd_files):
        basename = os.path.basename(fwd_path).replace('.pth', '')
        meta_path = fwd_path.replace('.pth', '_meta.json')
        
        latent_dim = 128
        arch = 'continuous'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            latent_dim = meta.get('latentDim', 128)
            arch = meta.get('pipelineArchitecture', 'continuous')
            
        f_state = torch.load(fwd_path, map_location=device, weights_only=True)
        if 'transition_model.0.weight' in f_state:
            hidden_dim = f_state['transition_model.0.weight'].shape[0]
        else:
            hidden_dim = 256
            
        if arch == 'discrete' and DiscreteLatentSLAM is not None:
            fwd_net = DiscreteLatentSLAM(latent_dim=latent_dim, model_size='large', image_size=64, num_actions=3, hidden_dim=hidden_dim).to(device)
        else:
            fwd_net = LatentSLAM(latent_dim=latent_dim, model_size='large', image_size=64, num_actions=3, hidden_dim=hidden_dim).to(device)
            
        fwd_net.load_state_dict(f_state, strict=False)
        fwd_net.eval()
        
        short_name = basename.split('_')[-1] if '_' in basename else basename[-6:]
        if short_name == "best" or short_name == "final":
            short_name = basename.split('_')[-2] + "_" + short_name
            
        oracles.append({
            'name': f"Neural Oracle ({short_name})",
            'uid': f"Neural_{idx}",
            'is_neural': True,
            'fwd_net': fwd_net,
            'ts_map': global_ts_map
        })
        
    print(f"Mounted {len(oracles)} Evaluators.")
    
    TEST_COUNT_PER_HORIZON = 400
    horizons = list(range(1, 11))
    
    # Store aggregated format
    metrics = {o['uid']: [] for o in oracles}
    metrics['Random'] = []
    
    for h in horizons:
        print(f"Evaluating Horizon: {h} steps")
        test_batch = []
        timeout = 0
        while len(test_batch) < TEST_COUNT_PER_HORIZON and timeout < 5000:
            timeout += 1
            traj = random.choice(trajectories)
            if len(traj) <= h + 1: continue
            
            idx1 = random.randint(0, len(traj) - h - 1)
            idx2 = idx1 + h
            
            truth_action = traj[idx1].get('macro_action', 0)
            if truth_action not in [1, 3, 4]: continue
                
            ts1 = traj[idx1]['lookup_ts']
            ts2 = traj[idx2]['lookup_ts']
            
            valid = True
            for o in oracles:
                if o.get('is_neural') and (ts1 not in o['ts_map'] or ts2 not in o['ts_map']):
                    valid = False
            if not valid: continue
            
            test_batch.append((ts1, ts2, truth_action))
            
        if len(test_batch) == 0:
            print(f"Warning: Could not find valid trajectories for horizon {h}")
            continue
            
        # Evaluate batch
        correct_counts = {o['uid']: 0 for o in oracles}
        correct_counts['Random'] = 0
        
        for (ts1, ts2, truth_action) in test_batch:
            t_cur = telemetry_dict[ts1]
            t_goal = telemetry_dict[ts2]
            
            r_act = random.choice([1, 3, 4])
            if r_act == truth_action: correct_counts['Random'] += 1
                
            for o in oracles:
                if not o['is_neural']:
                    a_act = algorithmic_math(t_cur, t_goal)
                    if a_act == 5: a_act = random.choice([1, 3, 4])
                    if a_act == truth_action: correct_counts[o['uid']] += 1
                else:
                    l_cur = o['ts_map'][ts1].to(device)
                    l_goal = o['ts_map'][ts2].to(device)
                    n_act = neural_logic(o['fwd_net'], None, l_cur, l_goal, device, threshold=3.0, is_rnn=False, cached_latents=None)
                    if n_act == 5: n_act = random.choice([1, 3, 4])
                    if n_act == truth_action: correct_counts[o['uid']] += 1
                    
        total_eval = len(test_batch)
        metrics['Random'].append((correct_counts['Random'] / total_eval) * 100.0)
        for o in oracles:
            metrics[o['uid']].append((correct_counts[o['uid']] / total_eval) * 100.0)

    # Plotting line chart
    plt.figure(figsize=(10, 6))
    plt.style.use('default')
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    
    # Plot random baseline
    plt.plot(horizons, metrics['Random'], label='Random Control', color='#64748b', marker='o', linewidth=2, linestyle='--')
    
    for o in oracles:
        if not o['is_neural']:
            color = '#f59e0b'
        else:
            color = '#10b981' if 'Neural_0' in o['uid'] else '#3b82f6'
            
        plt.plot(horizons, metrics[o['uid']], label=o['name'], color=color, marker='s', linewidth=2)
        
    plt.title("Action Prediction Accuracy by Trajectory Horizon Length", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Horizon Length (Number of Steps)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(horizons)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.5, color='gray')
    
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    for text in legend.get_texts(): text.set_color("black")
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "accuracy_vs_horizon.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Chart saved natively to {out_path}")

if __name__ == '__main__':
    main()
