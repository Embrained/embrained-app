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
    fwd_files = fwd_files[:2] # Limit to 2 for cleaner plotting
    
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
    
    # 1. Manually insert the Algorithmic Oracle into our dict logic for easy looping
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
            fwd_net.load_state_dict(f_state, strict=False)
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
    
    TEST_COUNT = 1500
    test_batch = []
    while len(test_batch) < TEST_COUNT:
        traj = random.choice(trajectories)
        idx1 = random.randint(0, len(traj) - 4)
        step_sz = random.randint(1, 3)
        idx2 = idx1 + step_sz
        
        truth_action = traj[idx1].get('macro_action', 0)
        if truth_action not in [1, 3, 4]:
            continue
            
        ts1 = traj[idx1]['lookup_ts']
        ts2 = traj[idx2]['lookup_ts']
        test_batch.append((ts1, ts2, truth_action))
        
    # Store true vs pred for each
    predictions = {o['uid']: {'y_true': [], 'y_pred': []} for o in oracles}
    
    print("Evaluating Test Batch...")
    for i, (ts1, ts2, truth_action) in enumerate(test_batch):
        if i % 100 == 0: print(f"Progress: {i}/{TEST_COUNT}")
        
        t_cur = telemetry_dict[ts1]
        t_goal = telemetry_dict[ts2]
        
        # Check neural validity first so we keep counts perfectly identical across grids
        neural_valid = True
        for o in oracles:
            if o['is_neural'] and (ts1 not in o['ts_map'] or ts2 not in o['ts_map']):
                neural_valid = False
                break
                
        if not neural_valid: continue
                
        for o in oracles:
            if not o['is_neural']:
                # Algorithmic calculation
                a_act = algorithmic_math(t_cur, t_goal)
                if a_act == 5: a_act = random.choice([1, 3, 4]) # Map Stop to random driving for pure path finding confusion
                predictions[o['uid']]['y_true'].append(truth_action)
                predictions[o['uid']]['y_pred'].append(a_act)
            else:
                # Neural calculation
                l_cur = o['ts_map'][ts1].to(device)
                l_goal = o['ts_map'][ts2].to(device)
                n_act = neural_logic(o['fwd_net'], None, l_cur, l_goal, device, threshold=3.0, is_rnn=False, cached_latents=None)
                if n_act == 5: n_act = random.choice([1, 3, 4])
                predictions[o['uid']]['y_true'].append(truth_action)
                predictions[o['uid']]['y_pred'].append(n_act)

    valid_cases = len(predictions[oracles[0]['uid']]['y_true'])
    print(f"Evaluation Complete. Formulated results on {valid_cases} instances.")
    
    # Plotting
    action_labels = [1, 3, 4]
    action_names = ['Forward', 'Left', 'Right']
    
    num_plots = len(oracles)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1: axes = [axes]
    
    fig.patch.set_facecolor('#1e1e1e')
    
    for ax, o in zip(axes, oracles):
        uid = o['uid']
        y_true = predictions[uid]['y_true']
        y_pred = predictions[uid]['y_pred']
        
        cm = confusion_matrix(y_true, y_pred, labels=action_labels)
        
        # Normalize by row (true label) to get % accuracy within that action class
        cm_norm = confusion_matrix(y_true, y_pred, labels=action_labels, normalize='true')
        
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=action_names, yticklabels=action_names,
                    cbar=False, annot_kws={"size": 14, "weight": "bold"})
                    
        ax.set_facecolor('#1e1e1e')
        ax.set_title(o['name'], color='white', size=14, pad=15)
        ax.set_ylabel('True Path Taken', color='white', size=12)
        ax.set_xlabel('Predicted Action', color='white', size=12)
        ax.tick_params(colors='white')
        
    plt.suptitle("Confusion Matrix: Algorithmic Ground-Truth vs Neural Predictions", color='white', size=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    out_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Chart saved natively to {out_path}")

if __name__ == '__main__':
    main()
