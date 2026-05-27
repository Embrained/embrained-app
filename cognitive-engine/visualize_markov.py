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
import pandas as pd
import cv2
import argparse
import numpy as np
import imageio
from scripts.extract_telemetry import TelemetryExtractor

def get_color(phase):
    if phase == 'Dwell': return (0, 255, 0) # Green
    elif phase == 'Randomizer': return (0, 165, 255) # Orange in BGR
    else: return (211, 0, 148) # Purple in BGR

def find_centroid_goal_image(input_dir):
    try:
        import glob
        import json
        import torch
        from PIL import Image
        import torchvision.transforms as T
        
        data_root = os.path.abspath(os.path.join(input_dir, ".."))
        goals_dir = os.path.join(data_root, "goals")
        
        with open(os.path.join(goals_dir, "group_stats.json"), 'r') as f:
            centroid = np.array(json.load(f)['centroid'])
            
        model_path = os.path.join(data_root, "vqvae_512c_32d_20260428_130632.pth")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        from backend.models.quantized_spatial import DiscreteLatentSLAM
        model = DiscreteLatentSLAM(latent_dim=32, num_actions=5, hidden_dim=256, model_size='large', image_size=64)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        min_dist = float('inf')
        closest_img = None
        
        for img_path in glob.glob(os.path.join(goals_dir, "frame_*.jpg")):
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                latent = model.encoder(tensor)
                latent = model.fc_e(latent)
                latent = latent.numpy().squeeze()
            
            dist = np.linalg.norm(latent - centroid)
            if dist < min_dist:
                min_dist = dist
                closest_img = img_path
                
        return closest_img
    except Exception as e:
        print(f"Warning: Failed to auto-detect centroid goal image: {e}")
        return None

def create_visualization(input_dir, output_name, fps=5, goal_image_path=None, disable_goal=False):
    csv_path = os.path.join(input_dir, "episode_data.csv")
    img_dir = os.path.join(input_dir, "images")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    output_path = os.path.join(input_dir, output_name)
    
    # Initialize Telemetry
    print("Initializing Telemetry Extractor...")
    extractor = TelemetryExtractor([])
    extractor.load_cache('telemetry_cache.npz')
    extractor._precompute_rotations()
    
    frames_for_bg_gray = []
    frames_for_bg_color = []
    sample_files = df['image_file'].sample(n=min(20, len(df)), random_state=42)
    for f in sample_files:
        f_name = str(f)
        if not f_name.startswith('webcam_'):
            f_name = 'webcam_' + f_name
        path = os.path.join(img_dir, f_name)
        if not os.path.exists(path):
            path = os.path.join(img_dir, str(f).replace('webcam_', ''))
        if os.path.exists(path):
            img_color = cv2.imread(path)
            if img_color is not None:
                if img_color.shape[:2] != (480, 640):
                    img_color = cv2.resize(img_color, (640, 480))
                frames_for_bg_color.append(img_color)
                frames_for_bg_gray.append(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY))
                
    bg_bgr = None
    if frames_for_bg_gray:
        extractor.initialize_moving_background(frames_for_bg_gray)
        bg_bgr = np.median(frames_for_bg_color, axis=0).astype(np.uint8)

    if not disable_goal and goal_image_path is None:
        goal_image_path = find_centroid_goal_image(input_dir)
        if goal_image_path:
            print(f"Auto-detected centroid goal image: {goal_image_path}")

    goal_img = None
    if not disable_goal and goal_image_path and os.path.exists(goal_image_path):
        goal_img = cv2.imread(goal_image_path)
        if goal_img is not None:
            goal_img = cv2.resize(goal_img, (160, 120))

    # Detect Dwells and Phases, offsetting by 1 to let inertia settle
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

    writer = None
    
    print(f"Processing {len(df)} transitions for {output_name}...")
    history = []
    
    for i in range(len(df)):
        if i % 10 == 0:
            print(f"Processed {i}/{len(df)} frames...")
            
        if phases[i] in ('Randomizer', 'Skip'):
            if i > 0 and phases[i-1] not in ('Randomizer', 'Skip'):
                history.clear()
                if bg_bgr is not None:
                    if writer is None:
                        writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
                    writer.append_data(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB))
            continue
            
        row = df.iloc[i]
        
        # Load Images
        img_file = str(row.get('image_file', f'frame_{i}.jpg'))
        
        pov_path = os.path.join(img_dir, img_file)
        if img_file.startswith('webcam_'):
            overhead_path = pov_path
            pov_path = os.path.join(img_dir, img_file.replace('webcam_', ''))
        else:
            overhead_path = os.path.join(img_dir, 'webcam_' + img_file)
            
        if not os.path.exists(overhead_path) or not os.path.exists(pov_path):
            continue
            
        overhead_frame = cv2.imread(overhead_path)
        pov_frame = cv2.imread(pov_path)
        
        if overhead_frame is None or pov_frame is None:
            continue
            
        if overhead_frame.shape != (480, 640, 3):
            overhead_frame = cv2.resize(overhead_frame, (640, 480))
            
        if writer is None:
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

        # Extract Telemetry
        overhead_gray = cv2.cvtColor(overhead_frame, cv2.COLOR_BGR2GRAY)
        feats = extractor.process_single_frame(overhead_gray)
        if feats is not None:
            cx, cy = feats['raw_cx'], feats['raw_cy']
            theta = np.degrees(np.arctan2(feats['sin_yaw'], feats['cos_yaw']))
            rad = np.deg2rad(theta)
            history.append((cx, cy, phases[i], rad))
            if len(history) > 50:
                history.pop(0)
                
        # Draw History Trace
        if len(history) > 1:
            for j in range(1, len(history)):
                pt1 = (int(history[j-1][0]), int(history[j-1][1]))
                pt2 = (int(history[j][0]), int(history[j][1]))
                color = get_color(history[j][2])
                cv2.line(overhead_frame, pt1, pt2, color, 1)
                
        for j, (hx, hy, hphase, hrad) in enumerate(history):
            color = get_color(hphase)
            r = 3 if j < len(history)-1 else 6
            center = (int(hx), int(hy))
            cv2.circle(overhead_frame, center, r, color, -1)
            
            # Draw orientation arrow
            arrow_len = 10 if j < len(history)-1 else 20
            dx = int(np.cos(hrad) * arrow_len)
            dy = int(np.sin(hrad) * arrow_len)
            end_pt = (center[0] + dx, center[1] + dy)
            thickness = 1 if j < len(history)-1 else 2
            cv2.arrowedLine(overhead_frame, center, end_pt, color, thickness, tipLength=0.3)

        # Create Inset (POV frame at bottom right)
        inset_w = 160
        inset_h = 120
        pov_resized = cv2.resize(pov_frame, (inset_w, inset_h))
        
        h, w = overhead_frame.shape[:2]
        bw = 4 # border width
        
        if goal_img is not None:
            box_w = (2 * inset_w) + (3 * bw)
            box_h = inset_h + (2 * bw)
            box_x = w - box_w
            box_y = h - box_h
            
            # Draw unified black background
            cv2.rectangle(overhead_frame, (box_x, box_y), (w, h), (0, 0, 0), -1)
            
            # Place Goal Image
            goal_x = box_x + bw
            goal_y = box_y + bw
            overhead_frame[goal_y : goal_y + inset_h, goal_x : goal_x + inset_w] = goal_img
            cv2.putText(overhead_frame, "Goal", (goal_x + 5, goal_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Place POV Image
            pov_x = goal_x + inset_w + bw
            pov_y = box_y + bw
            overhead_frame[pov_y : pov_y + inset_h, pov_x : pov_x + inset_w] = pov_resized
            cv2.putText(overhead_frame, "POV", (pov_x + 5, pov_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            box_w = inset_w + (2 * bw)
            box_h = inset_h + (2 * bw)
            box_x = w - box_w
            box_y = h - box_h
            
            # Draw unified black background
            cv2.rectangle(overhead_frame, (box_x, box_y), (w, h), (0, 0, 0), -1)
            
            # Place POV Image
            pov_x = box_x + bw
            pov_y = box_y + bw
            overhead_frame[pov_y : pov_y + inset_h, pov_x : pov_x + inset_w] = pov_resized
            cv2.putText(overhead_frame, "POV", (pov_x + 5, pov_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw Text
        l_val = float(row.get('pwm_left', 0.0))
        r_val = float(row.get('pwm_right', 0.0))
        txt1 = f"Action: PWM: L={int(l_val)} R={int(r_val)}"
        cv2.putText(overhead_frame, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        canvas_rgb = cv2.cvtColor(overhead_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(canvas_rgb)
            
    if writer:
        writer.close()
        print(f"Successfully created visualization video at:\n{output_path}")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Markov Transition Visualization")
    parser.add_argument("directory", help="Path to markov_ directory")
    parser.add_argument("--out", default="markov_visualization.mp4", help="Output filename")
    parser.add_argument("--goal", default=None, help="Path to goal image to display in an inset")
    parser.add_argument("--nogoalimg", action="store_true", help="Disable the goal image inset")
    args = parser.parse_args()
    
    create_visualization(args.directory, args.out, goal_image_path=args.goal, disable_goal=args.nogoalimg)
