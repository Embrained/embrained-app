import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scripts.extract_telemetry import TelemetryExtractor

def plot_trajectory(dataset_dir, label=None):
    print(f"Loading Extractor for {dataset_dir}...")
    extractor = TelemetryExtractor([])
    extractor.load_cache('telemetry_cache.npz')
    extractor._precompute_rotations()
    
    csv_path = os.path.join(dataset_dir, 'episode_data.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    print("Initializing background...")
    frames = []
    sample_files = df['image_file'].sample(n=min(20, len(df)), random_state=42)
    for f in sample_files:
        if not f.startswith('webcam_'):
            f = 'webcam_' + f
        path = os.path.join(dataset_dir, 'images', f)
        if not os.path.exists(path):
            path = os.path.join(dataset_dir, 'images', f.replace('webcam_', ''))
        if os.path.exists(path):
            img = cv2.imread(path, 0)
            if img is not None:
                if img.shape != (480, 640):
                    img = cv2.resize(img, (640, 480))
                frames.append(img)
    if frames:
        extractor.initialize_moving_background(frames)
    
    # 1. Detect Dwells (>=5 consecutive stops), offsetting by 1 to let inertia settle
    is_zero = (df['pwm_left'] == 0) & (df['pwm_right'] == 0)
    is_dwell = [False] * len(df)
    block_start = 0
    curr_z = is_zero.iloc[0]
    for i in range(len(df)):
        z = is_zero.iloc[i]
        if z != curr_z:
            length = i - block_start
            if curr_z and length >= 5:
                for j in range(block_start + 1, i):
                    is_dwell[j] = True
            curr_z = z
            block_start = i
    if curr_z and (len(df) - block_start) >= 5:
        for j in range(block_start + 1, len(df)):
            is_dwell[j] = True

    # 2. Extract telemetry and assign phases
    phases = []
    moving_state = 'Trial'
    last_was_dwell = is_dwell[0]
    for i in range(len(df)):
        if is_dwell[i]:
            phases.append('Dwell')
            if not last_was_dwell:
                moving_state = 'Randomizer' if moving_state == 'Trial' else 'Trial'
            last_was_dwell = True
        else:
            last_was_dwell = False
            phases.append(moving_state)

    # Filter out Warmup Dwells, 1-step trials, and Randomizers
    blocks = []
    block_start = 0
    current_phase = phases[0]
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != current_phase:
            blocks.append({'phase': current_phase, 'start': block_start, 'end': i, 'len': i - block_start})
            if i < len(phases):
                current_phase = phases[i]
                block_start = i

    for idx, block in enumerate(blocks):
        if block['phase'] == 'Randomizer':
            for i in range(block['start'], block['end']):
                phases[i] = 'Skip'
                
        elif block['phase'] == 'Dwell':
            prev_phase = 'Randomizer'
            for j in range(idx - 1, -1, -1):
                if blocks[j]['phase'] != 'Dwell':
                    prev_phase = blocks[j]['phase']
                    break
            if prev_phase == 'Randomizer':
                for i in range(block['start'], block['end']):
                    phases[i] = 'Skip'
                    
        elif block['phase'] == 'Trial':
            if block['len'] == 1:
                for i in range(block['start'], block['end']):
                    phases[i] = 'Skip'
                for j in range(idx + 1, len(blocks)):
                    if blocks[j]['phase'] == 'Dwell':
                        for k in range(blocks[j]['start'], blocks[j]['end']):
                            phases[k] = 'Skip'
                        break
                    elif blocks[j]['phase'] != 'Dwell':
                        break

    xs = {'Trial': [], 'Randomizer': [], 'Dwell': []}
    ys = {'Trial': [], 'Randomizer': [], 'Dwell': []}
    us = {'Trial': [], 'Randomizer': [], 'Dwell': []}
    vs = {'Trial': [], 'Randomizer': [], 'Dwell': []}
    
    for i in range(len(df)):
        phase = phases[i]
        if phase == 'Skip':
            continue
            
        row = df.iloc[i]
        img_file = row['image_file']
        if not img_file.startswith('webcam_'):
            img_file = 'webcam_' + img_file
        img_path = os.path.join(dataset_dir, 'images', img_file)
        
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_dir, 'images', row['image_file'])
            if not os.path.exists(img_path):
                continue
                
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if img_gray.shape != (480, 640):
            img_gray = cv2.resize(img_gray, (640, 480))
            
        feats = extractor.process_single_frame(img_gray)
        if feats is None:
            continue
            
        cx, cy = feats['raw_cx'], feats['raw_cy']
        theta = np.degrees(np.arctan2(feats['sin_yaw'], feats['cos_yaw']))
        rad = np.deg2rad(theta)
        u = np.cos(rad) * 20
        v = np.sin(rad) * 20
        
        # Determine Phase
        phase = 'Trial'
        if is_dwell[i]:
            phase = 'Dwell'
            if not last_was_dwell:
                moving_state = 'Randomizer' if moving_state == 'Trial' else 'Trial'
            last_was_dwell = True
        else:
            last_was_dwell = False
            phase = moving_state
                
        xs[phase].append(cx)
        ys[phase].append(cy)
        us[phase].append(u)
        vs[phase].append(v)
        
    # Plotting
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.invert_yaxis()
    
    # Trial (blue/purple)
    if xs['Trial']:
        plt.quiver(xs['Trial'], ys['Trial'], us['Trial'], vs['Trial'], color='mediumpurple', alpha=0.9, angles='xy', scale_units='xy', scale=1, label='Trial Steps', width=0.004)
        plt.scatter(xs['Trial'], ys['Trial'], color='mediumpurple', alpha=0.9, s=20)
        
    # Randomizer (orange)
    if xs['Randomizer']:
        plt.quiver(xs['Randomizer'], ys['Randomizer'], us['Randomizer'], vs['Randomizer'], color='orange', alpha=0.7, angles='xy', scale_units='xy', scale=1, label='Randomizer Steps', width=0.003)
        plt.scatter(xs['Randomizer'], ys['Randomizer'], color='orange', alpha=0.7, s=15)
        
    # Dwell (green)
    if xs['Dwell']:
        # Dwells don't move much, arrows might clutter, but we'll plot them
        plt.quiver(xs['Dwell'], ys['Dwell'], us['Dwell'], vs['Dwell'], color='limegreen', alpha=0.5, angles='xy', scale_units='xy', scale=1, label='Dwell Phase', width=0.002)
        plt.scatter(xs['Dwell'], ys['Dwell'], color='limegreen', alpha=0.5, s=10)
        
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    title_str = f"Trajectory Visualization: {dataset_name}"
    if label:
        title_str += f" ({label.capitalize()} Controller)"
    title_str += "\nTelemetry from Extractor"
    plt.title(title_str)
    
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure reasonable limits if no data
    all_xs = xs['Trial'] + xs['Randomizer'] + xs['Dwell']
    all_ys = ys['Trial'] + ys['Randomizer'] + ys['Dwell']
    if all_xs:
        plt.xlim(min(all_xs)-50, max(all_xs)+50)
        plt.ylim(max(all_ys)+50, min(all_ys)-50) # inverted y
        
    output_path = os.path.join(dataset_dir, f"{dataset_name}_trajectory.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to markov_ directory")
    parser.add_argument("--label", help="Controller label for title (e.g., neural, telemetry, random)")
    args = parser.parse_args()
    
    plot_trajectory(args.directory, args.label)
