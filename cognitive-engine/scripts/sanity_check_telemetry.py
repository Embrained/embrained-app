import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def angular_distance(yaw1, yaw2):
    diff = (yaw2 - yaw1) % 360
    if diff > 180:
        diff -= 360
    return diff

def main():
    tel_path = "data/master_telemetry.csv"
    trans_path = "data/all_transitions.json"
    
    if not os.path.exists(tel_path):
        print(f"Error: {tel_path} not found.")
        return
        
    print(f"Loading telemetry from {tel_path}...")
    df_tel = pd.read_csv(tel_path)
    
    if not os.path.exists(trans_path):
        print(f"Error: {trans_path} not found. Needs transitions to map action IDs.")
        return
        
    print(f"Loading transitions from {trans_path}...")
    with open(trans_path, "r") as f:
        transitions = json.load(f)

    # 1. Merge matching timestamps
    print("Extracting action telemetry...")
    tel_map = {str(int(row['ts'])): row for _, row in df_tel.iterrows()}
    
    data = []
    action_names = {1: 'FWD', 2: 'REV', 3: 'HW_LEFT', 4: 'HW_RIGHT', 5: 'SW_LEFT', 6: 'SW_RIGHT', 0: 'STOP'}
    
    for i in range(len(transitions) - 1):
        curr_t = transitions[i]
        next_t = transitions[i+1]
        
        # Ensure contiguous
        if curr_t.get('session') != next_t.get('session'):
            continue
            
        action = curr_t.get('macro_action', 0)
        
        ts_curr = str(int(curr_t['timestamp'] * 1000))
        ts_next = str(int(next_t['timestamp'] * 1000))
        
        if ts_curr in tel_map and ts_next in tel_map:
            row_curr = tel_map[ts_curr]
            row_next = tel_map[ts_next]
            
            dx = row_next['cx'] - row_curr['cx']
            dy = row_next['cy'] - row_curr['cy']
            dist = np.sqrt(dx**2 + dy**2)
            dyaw = angular_distance(row_curr['yaw_deg'], row_next['yaw_deg'])
            
            # Extract folder from img_dir
            # E.g. ".../data/markov_2026-03-22_13-34-29/images"
            # normalized path with slashes
            img_dir_norm = str(row_curr['img_dir']).replace("\\", "/")
            folder = os.path.basename(os.path.dirname(img_dir_norm))
            
            data.append({
                'folder': folder,
                'action_id': action,
                'action_name': action_names.get(action, f"ACT_{action}"),
                'dist_px': dist,
                'dyaw_deg': dyaw
            })

    if not data:
        print("No valid telemetry-action pairs found.")
        return
        
    df_res = pd.DataFrame(data)
    
    # We focus on the big movement actions usually
    df_res = df_res[df_res['action_id'].isin([1, 2, 3, 4])]
    
    print(f"Extracted {len(df_res)} valid action transitions.")

    # 2. Pool statistics
    pool_stats = df_res.groupby('action_name', as_index=False).agg(
        pool_dist_mean=('dist_px', 'mean'),
        pool_dist_std=('dist_px', 'std'),
        pool_yaw_mean=('dyaw_deg', 'mean'),
        pool_yaw_std=('dyaw_deg', 'std')
    )
    
    print("\n" + "="*50)
    print(" POOLED ACTION-SPECIFIC TELEMETRY")
    print("="*50)
    print(pool_stats.to_string(index=False))
    
    # 3. Folder statistics
    folder_stats = df_res.groupby(['folder', 'action_name']).agg(
        count=('dist_px', 'count'),
        f_dist_mean=('dist_px', 'mean'),
        f_yaw_mean=('dyaw_deg', 'mean')
    ).reset_index()
    
    # 4. Outlier Analysis
    # Compare each folder to pool stats
    folder_stats = folder_stats.merge(pool_stats, on='action_name')
    
    folder_stats['pool_dist_std'] = folder_stats['pool_dist_std'].replace(0, np.nan)
    folder_stats['pool_yaw_std'] = folder_stats['pool_yaw_std'].replace(0, np.nan)

    # Calculate absolute Z-scores
    folder_stats['z_dist'] = np.abs(folder_stats['f_dist_mean'] - folder_stats['pool_dist_mean']) / folder_stats['pool_dist_std']
    folder_stats['z_yaw'] = np.abs(folder_stats['f_yaw_mean'] - folder_stats['pool_yaw_mean']) / folder_stats['pool_yaw_std']
    
    Z_THRESH = 3.0
    MIN_SAMPLES = 5 # Avoid flagging recordings with just 1 or 2 steps for that action
    
    ood_dist = folder_stats[(folder_stats['count'] >= MIN_SAMPLES) & (folder_stats['z_dist'] > Z_THRESH)].sort_values('z_dist', ascending=False)
    ood_yaw = folder_stats[(folder_stats['count'] >= MIN_SAMPLES) & (folder_stats['z_yaw'] > Z_THRESH)].sort_values('z_yaw', ascending=False)
    
    print("\n" + "="*50)
    print(f" OUT-OF-DISTRIBUTION RECORDINGS (Z > {Z_THRESH})")
    print("="*50)
    
    print("\n[== DISTANCE OUTLIERS ==]")
    if len(ood_dist) > 0:
        for _, r in ood_dist.iterrows():
            print(f"-> {r['folder']} | {r['action_name']}")
            print(f"   Samples: {r['count']}, Z-Score: {r['z_dist']:.2f}")
            print(f"   Folder Mean: {r['f_dist_mean']:.2f} px | Pool Mean: {r['pool_dist_mean']:.2f} px")
    else:
        print("No significant distance anomalies found.")
        
    print("\n[== YAW OUTLIERS ==]")
    if len(ood_yaw) > 0:
        for _, r in ood_yaw.iterrows():
            print(f"-> {r['folder']} | {r['action_name']}")
            print(f"   Samples: {r['count']}, Z-Score: {r['z_yaw']:.2f}")
            print(f"   Folder Mean: {r['f_yaw_mean']:.2f} deg | Pool Mean: {r['pool_yaw_mean']:.2f} deg")
    else:
        print("No significant yaw anomalies found.")

    print("\nGenerating visualization...")
    
    actions = ['FWD', 'REV', 'HW_LEFT', 'HW_RIGHT']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Recording Folders Empirical Action Scatter', fontsize=16)
    
    for ax, action in zip(axes.flatten(), actions):
        ax.set_title(f'Action: {action}')
        ax.set_xlabel('Mean Delta Yaw (degrees)')
        ax.set_ylabel('Mean Distance (px)')
        
        # Pool stats
        p_stat = pool_stats[pool_stats['action_name'] == action]
        if p_stat.empty:
            continue
            
        p_dist = p_stat['pool_dist_mean'].values[0]
        p_dist_s = p_stat['pool_dist_std'].values[0]
        p_yaw = p_stat['pool_yaw_mean'].values[0]
        p_yaw_s = p_stat['pool_yaw_std'].values[0]
        
        # Draw pool center
        ax.plot(p_yaw, p_dist, 'rX', markersize=12, label='Pooled Average', zorder=5)
        
        # 3 std dev bounds
        min_yaw = p_yaw - Z_THRESH * p_yaw_s
        max_yaw = p_yaw + Z_THRESH * p_yaw_s
        min_dist = p_dist - Z_THRESH * p_dist_s
        max_dist = p_dist + Z_THRESH * p_dist_s
        
        # Draw bounding rectangle
        bx = [min_yaw, max_yaw, max_yaw, min_yaw, min_yaw]
        by = [min_dist, min_dist, max_dist, max_dist, min_dist]
        ax.plot(bx, by, 'r--', label=f'{Z_THRESH} Std Dev Bounds')
        
        # Plot individual folders
        f_stat = folder_stats[(folder_stats['action_name'] == action) & (folder_stats['count'] >= MIN_SAMPLES)]
        
        if f_stat.empty:
            continue
            
        # Split into normal and outlier
        normal_f = f_stat[(f_stat['z_dist'] <= Z_THRESH) & (f_stat['z_yaw'] <= Z_THRESH)]
        outlier_f = f_stat[(f_stat['z_dist'] > Z_THRESH) | (f_stat['z_yaw'] > Z_THRESH)]
        
        ax.scatter(normal_f['f_yaw_mean'], normal_f['f_dist_mean'], c='blue', alpha=0.6, label='In-Dist Folder')
        if not outlier_f.empty:
            ax.scatter(outlier_f['f_yaw_mean'], outlier_f['f_dist_mean'], c='red', alpha=0.9, marker='D', label='Outlier Folder')
            
            # Label outliers
            for _, r in outlier_f.iterrows():
                short_name = r['folder'].split('_')[-1]
                ax.annotate(short_name, (r['f_yaw_mean'], r['f_dist_mean']), xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
                
        ax.grid(alpha=0.3)
        ax.legend(prop={'size': 8})
        
    plt.tight_layout()
    out_file = "sanity_telemetry_outliers.png"
    plt.savefig(out_file, dpi=150)
    print(f"Visualization saved to {out_file}")

    print("\nDone.")

if __name__ == "__main__":
    main()
