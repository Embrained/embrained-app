import os
import sys
import json
import glob
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    fwd_files = fwd_files[:3]
    
    cache_files = glob.glob(os.path.join(data_dir, "*-vae_*_global_latents.pt"))
    cache_files.sort(key=os.path.getmtime, reverse=True)
    
    if not cache_files:
        print("ERROR: No global latents cache found in data directory.")
        return
        
    cache_path = cache_files[0]
    print(f"Using Global Latents Cache: {os.path.basename(cache_path)}")
    latent_data = torch.load(cache_path, map_location='cpu', weights_only=True)
    global_ts_map = latent_data.get("ts_map", {})
    
    oracles = []
    for fwd_path in fwd_files:
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
            fwd_net.load_state_dict(f_state, strict=False)
        else:
            fwd_net = LatentSLAM(latent_dim=latent_dim, model_size='large', image_size=64, num_actions=3, hidden_dim=hidden_dim).to(device)
            fwd_net.load_state_dict(f_state, strict=False)
            
        fwd_net.eval()
        
        # Clean up the name for the chart
        # "topological_forward_latentslam_20260416_123614" -> "Oracle: 123614"
        short_name = basename.split('_')[-1] if '_' in basename else basename[-6:]
        if short_name == "best" or short_name == "final":
            short_name = basename.split('_')[-2] + "_" + short_name
            
        oracles.append({
            'name': f"Neural Oracle ({short_name})",
            'fwd_net': fwd_net,
            'ts_map': global_ts_map
        })
        
    print(f"Mounted {len(oracles)} Neural Oracles for Evaluation.")
    
    TEST_COUNT = 1000
    test_batch = []
    # Try to pick trajectories of length 1, 2, or 3
    while len(test_batch) < TEST_COUNT:
        traj = random.choice(trajectories)
        idx1 = random.randint(0, len(traj) - 4)
        step_sz = random.randint(1, 3)
        idx2 = idx1 + step_sz
        
        # Action taken at idx1:
        truth_action = traj[idx1].get('macro_action', 0)
        # Only evaluate valid driving actions FWD=1, L=3, R=4
        if truth_action not in [1, 3, 4]:
            continue
            
        ts1 = traj[idx1]['lookup_ts']
        ts2 = traj[idx2]['lookup_ts']
        test_batch.append((ts1, ts2, truth_action))
        
    # Stats
    correct_counts = {'Algorithmic': 0, 'Random': 0}
    # Ensure distinct tracking
    for idx, o in enumerate(oracles):
        o['uid'] = f"{o['name']}_{idx}"
        correct_counts[o['uid']] = 0
        
    valid_neural_tests = 0
        
    print("Evaluating Test Batch...")
    for i, (ts1, ts2, truth_action) in enumerate(test_batch):
        if i % 100 == 0: print(f"Progress: {i}/{TEST_COUNT}")
        
        t_cur = telemetry_dict[ts1]
        t_goal = telemetry_dict[ts2]
        
        # Algorithmic Oracle
        a_act = algorithmic_math(t_cur, t_goal)
        if a_act == truth_action:
            correct_counts['Algorithmic'] += 1
            
        # Random Control
        r_act = random.choice([1, 3, 4])
        if r_act == truth_action:
            correct_counts['Random'] += 1
            
        # Neural Oracles
        neural_valid = True
        for o in oracles:
            if ts1 not in o['ts_map'] or ts2 not in o['ts_map']:
                neural_valid = False
                break
                
        if neural_valid:
            valid_neural_tests += 1
            for o in oracles:
                l_cur = o['ts_map'][ts1].to(device)
                l_goal = o['ts_map'][ts2].to(device)
                
                n_act = neural_logic(o['fwd_net'], None, l_cur, l_goal, device, threshold=3.0, is_rnn=False, cached_latents=None)
                if n_act == truth_action:
                    correct_counts[o['uid']] += 1

    print(f"Evaluation Complete. Tested fully on {valid_neural_tests} cases.")
    
    # Calculate Percentages
    results_list = []
    results_list.append({"Oracle": "Algorithmic Ground Truth", "Accuracy (%)": (correct_counts['Algorithmic'] / TEST_COUNT) * 100.0})
    results_list.append({"Oracle": "Random Control", "Accuracy (%)": (correct_counts['Random'] / TEST_COUNT) * 100.0})
    
    for o in oracles:
        if valid_neural_tests > 0:
            pct = (correct_counts[o['uid']] / valid_neural_tests) * 100.0
        else:
            pct = 0.0
        results_list.append({"Oracle": o['name'], "Accuracy (%)": pct})
        
    df_results = pd.DataFrame(results_list)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    colors = []
    for oracle in df_results['Oracle']:
        if 'Random' in oracle: colors.append('#64748b')
        elif 'Algorithmic' in oracle: colors.append('#f59e0b')
        else: colors.append('#10b981')
        
    bars = plt.bar(df_results['Oracle'], df_results['Accuracy (%)'], color=colors)
    plt.title("Short-Horizon Accuracy: Predicting the True Path Taken", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Predictive Oracle", fontsize=12)
    plt.xticks(rotation=15, ha='right')
    
    # Add accuracy text
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12, color='white')

    plt.tight_layout()
    
    # Force dark background everywhere to avoid transparent text dropouts
    fig = plt.gcf()
    fig.patch.set_facecolor('#1e1e1e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    out_path = os.path.join(output_dir, "accuracy_evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
    plt.close()
    
    print(f"Chart saved natively to {out_path}")

if __name__ == '__main__':
    main()
