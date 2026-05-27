import os
import sys
import json
import glob
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.training.evaluate_oracles import algorithmic_math, neural_logic
from backend.models.latentslam import LatentSLAM
# Import DiscreteLatentSLAM if available
try:
    from backend.models.quantized_spatial import DiscreteLatentSLAM
except ImportError:
    DiscreteLatentSLAM = None
from modules.spatial_model import TinyVAE

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
    
    # Just grab the latest VAE global cache! 
    # Because Phase 1 -> Phase 2 is chronological, the latest global cache corresponds to the active embeddings.
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
        
        # Instantiate correct base size
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
        
        oracles.append({
            'name': f"Neural Prediction ({basename[-15:-7]})",
            'fwd_net': fwd_net,
            'ts_map': global_ts_map
        })
        
    print(f"Mounted {len(oracles)} Neural Oracles for Evaluation.")
    
    # Generate Test Batch
    TEST_COUNT = 1000
    test_batch = []
    for _ in range(TEST_COUNT):
        traj = random.choice(trajectories)
        idx1 = random.randint(0, len(traj) - 3)
        # Random step size between 1 and 5
        step_sz = random.randint(1, 5)
        idx2 = min(len(traj) - 1, idx1 + step_sz)
        
        ts1_prev = traj[idx1-1]['lookup_ts'] if idx1 > 0 else traj[idx1]['lookup_ts']
        ts1 = traj[idx1]['lookup_ts']
        ts2 = traj[idx2]['lookup_ts']
        test_batch.append((ts1_prev, ts1, ts2))
        
    # Tally Dictionary
    actions_map = {1: 'Forward', 3: 'Left', 4: 'Right', 5: 'Stop'}
    
    results = []
    
    algo_stops = 0
    # Process
    print("Evaluating Test Batch...")
    for i, (ts1_prev, ts1, ts2) in enumerate(test_batch):
        if i % 100 == 0: print(f"Progress: {i}/{TEST_COUNT}")
        
        t_cur = telemetry_dict[ts1]
        t_goal = telemetry_dict[ts2]
        
        # Algorithmic Oracle
        a_act = algorithmic_math(t_cur, t_goal)
        results.append({'Oracle': 'Algorithmic (Ground Truth)', 'Action': actions_map.get(a_act, 'Other')})
        if a_act == 5: algo_stops += 1
        
        # Random Control
        r_act = random.choice([1, 3, 4, 5])
        results.append({'Oracle': 'Random Control', 'Action': actions_map.get(r_act, 'Other')})
        
        # Neural Oracles
        for o in oracles:
            if ts1 not in o['ts_map'] or ts2 not in o['ts_map']:
                continue
            l_cur = o['ts_map'][ts1].to(device)
            l_goal = o['ts_map'][ts2].to(device)
            
            n_act = neural_logic(o['fwd_net'], None, l_cur, l_goal, device, threshold=3.0, is_rnn=False, cached_latents=None)
            results.append({'Oracle': o['name'], 'Action': actions_map.get(n_act, 'Other')})
            
    print(f"Evaluation Complete. Algorithm emitted {algo_stops} Stops.")
    
    # Plotting
    df_results = pd.DataFrame(results)
    
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    
    # Custom color palette matching the requested actions
    colors = {'Forward': '#3b82f6', 'Left': '#10b981', 'Right': '#f59e0b', 'Stop': '#ef4444', 'Other': '#64748b'}
    
    # Crosstab the percentages
    xtab = pd.crosstab(df_results['Oracle'], df_results['Action'], normalize='index') * 100
    ax = xtab.plot(kind='bar', stacked=True, color=[colors.get(c, '#64748b') for c in xtab.columns], figsize=(12, 8))
    
    plt.title("Action Distribution Across Short (1-5 step) Horizons", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Percentage Picked (%)", fontsize=12)
    plt.xlabel("Oracle Evaluator", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage text within bars
    for c in ax.containers:
        # Standardize bar labels
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontweight='bold', fontsize=10, color='white')

    plt.tight_layout()
    
    # Force dark background to prevent text washout on export
    fig = plt.gcf()
    fig.patch.set_facecolor('#1e1e1e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    legend = plt.legend(title='Predicted Action', bbox_to_anchor=(1.05, 1), loc='upper left')
    legend.get_frame().set_facecolor('#1e1e1e')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts(): text.set_color("white")
    legend.get_title().set_color("white")
    
    out_path = os.path.join(output_dir, "action_distribution_tally.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
    plt.close()
    
    print(f"Chart saved natively to {out_path}")

if __name__ == "__main__":
    main()
