import os
import sys
import glob
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Mount backend components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.models.latentslam import LatentSLAM

def run_short_horizon_analysis():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Initializing Short Horizon Reachability Analysis...")

    # Load dataset structures
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
        traj = sorted(sessions[s], key=lambda x: x.get('timestamp', 0))
        # Keep sequences that strictly use [1, 3, 4] for pure topological mapping
        valid_traj = []
        for n in traj:
            mac = n.get('macro_action', 0)
            if mac in [1, 3, 4]:
                valid_traj.append(n)
        
        if len(valid_traj) > 6:
            trajectories.append(valid_traj)

    print(f"Aggregated {len(trajectories)} robust topological trajectories.")

    # Locate Models
    f_candidates = glob.glob(os.path.join(data_root, 'topological_forward_latentslam_*.pth'))
    if not f_candidates:
        print("No LatentSLAM forward models found. Aborting.")
        return
    f_candidates.sort(key=os.path.getmtime, reverse=True)
    target_f_net = f_candidates[0]

    cache_candidates = glob.glob(os.path.join(data_root, '*_global_latents.pt'))
    if not cache_candidates:
         print("Global latents map missing. Re-extract VAE latents first.")
         return
    cache_candidates.sort(key=os.path.getmtime, reverse=True)
    cache_path = cache_candidates[0]
    
    # Load Latent dictionary mapping basename -> latent [32]
    latent_cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    if 'path_map' in latent_cache:
        lat_map = {os.path.basename(k): v for k, v in latent_cache['path_map'].items()}
    else:
        # Fallback to ts_map if path_map not found
        lat_map = latent_cache.get('ts_map', {})

    print(f"Loading LatentSLAM Simulator: {os.path.basename(target_f_net)}")
    f_state = torch.load(target_f_net, map_location=device, weights_only=True)
    if 'transition_model.0.weight' in f_state:
        hidden_dim = f_state['transition_model.0.weight'].shape[0]
        out_features = f_state['transition_model.4.weight'].shape[0] if 'transition_model.4.weight' in f_state else 32 * 3
        # Assume standard 3 actions if unspecified explicitly
        latent_dim = 32
        for k, v in f_state.items():
            if 'transition_model.0.weight' in k: latent_dim = v.shape[1]
        num_acts = out_features // latent_dim
    else:
        print("Model is not LatentSLAM structured. Aborting.")
        return

    f_net = LatentSLAM(latent_dim=latent_dim, num_actions=num_acts, hidden_dim=hidden_dim).to(device)
    f_net.load_state_dict(f_state, strict=False)
    f_net.eval()

    # Evaluation Params
    MAX_HORIZON = 5
    TESTS_PER_HORIZON = 200
    THRESHOLD_L2 = 3.0

    horizons = list(range(1, MAX_HORIZON + 1))
    reachability_scores = []
    parity_scores = []
    avg_l2_scores = []

    for H in horizons:
        print(f"\nEvaluating Validation Horizon: {H} Steps [{3**H} BFS Nodes]")
        
        successes = 0
        exact_matches = 0
        l2_accum = 0
        
        for t in range(TESTS_PER_HORIZON):
            # Sample trajectory
            traj = random.choice(trajectories)
            if len(traj) <= H + 1: continue
            
            # Sub-sample contiguous chunk
            start_idx = random.randint(0, len(traj) - H - 1)
            chunk = traj[start_idx:start_idx + H + 1]
            
            # Make sure timestamps exist in lat_map via direct lookup
            valid = True
            latents = []
            true_actions = []
            
            for i, node in enumerate(chunk):
                path = node.get('image_path', '')
                base = os.path.basename(path)
                ts = base.replace('frame_', '').replace('.jpg', '')
                
                # Check lat_map keys
                if base in lat_map: latents.append(lat_map[base].to(device))
                elif ts in lat_map: latents.append(lat_map[ts].to(device))
                else:
                    valid = False
                    break
                    
                if i < len(chunk) - 1:
                    a = chunk[i+1].get('macro_action', 0)
                    true_actions.append(a)
                    
            if not valid or len(latents) != H + 1:
                # If we drop a frame, loop retry
                continue
                
            l_start = latents[0]
            l_goal = latents[-1]
            
            # BFS 
            # state structure: (latent_tensor, action_sequence_history)
            queue = [(l_start, [])]
            for depth in range(H):
                next_queue = []
                for state_l, seq in queue:
                    # state_l is [latent_dim]
                    l_in = state_l.unsqueeze(0) # [1, latent_dim]
                    with torch.no_grad():
                        preds = f_net.predict_next_state(l_in) # [1, 3, latent_dim]
                        
                    for act in [1, 3, 4]:
                        act_idx = 0 if act == 1 else (1 if act == 3 else 2)
                        next_state = preds[0, act_idx] # [latent_dim]
                        next_queue.append((next_state, seq + [act]))
                        
                queue = next_queue
                
            # Leaves evaluation
            best_l2 = float('inf')
            best_seq = None
            
            for state_l, seq in queue:
                dist = torch.norm(state_l - l_goal, p=2).item()
                if dist < best_l2:
                    best_l2 = dist
                    best_seq = seq
                    
            l2_accum += best_l2
            if best_l2 < THRESHOLD_L2:
                successes += 1
                
            if best_seq == true_actions:
                exact_matches += 1
                
        reach_pct = (successes / TESTS_PER_HORIZON) * 100.0
        parity_pct = (exact_matches / TESTS_PER_HORIZON) * 100.0
        avg_l2 = l2_accum / TESTS_PER_HORIZON
        
        reachability_scores.append(reach_pct)
        parity_scores.append(parity_pct)
        avg_l2_scores.append(avg_l2)
        
        print(f" -> Reachability (L2 < {THRESHOLD_L2}): {reach_pct:.1f}%")
        print(f" -> Optimal Sequence Match: {parity_pct:.1f}%")
        print(f" -> Avg Lowest Terminal Distance: {avg_l2:.3f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    
    color = '#1f77b4'
    ax1.set_xlabel('Simulated Horizon (Steps Ahead)', fontsize=12, fontweight='bold', color='#333333')
    ax1.set_ylabel('Success Rate (%)', color=color, fontweight='bold', fontsize=12)
    ln1 = ax1.plot(horizons, reachability_scores, color=color, marker='o', linewidth=3, markersize=8, label='Spatial Reachability (L2 < 3.0)')
    ln2 = ax1.plot(horizons, parity_scores, color='#2ca02c', marker='s', linewidth=2, markersize=7, label='Optimal Action Path Parity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('Avg Minimal Terminal Euclidean Distance', color=color2, fontweight='bold', fontsize=12)
    ln3 = ax2.plot(horizons, avg_l2_scores, color=color2, marker='x', linestyle='--', linewidth=2, markersize=8, label='L2 Divergence')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(threshold_padding := max(avg_l2_scores) * 1.3, 5.0))
    
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', fontsize=11, framealpha=0.9)
    
    plt.title("LatentSLAM Geometric Simulation Decay", fontsize=16, fontweight='bold', color='#111111', pad=15)
    
    out_path = os.path.join(data_root, "short_horizon_metrics.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis complete. Visual rendering exported to: {out_path}")

if __name__ == '__main__':
    run_short_horizon_analysis()
