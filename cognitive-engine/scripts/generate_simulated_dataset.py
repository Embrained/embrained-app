import os
import sys
import argparse
import time
import random
import json
import math
import csv
import logging
from datetime import datetime
import numpy as np
import cv2

# Ensure we can import from software_suite root modular definitions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.sim.raycast.simulator import RaycastSimulator
from modules.sim.raycast.renderer import dist_to_sensor
from config import DATA_DIR, ACTION_PWM_MAP

def generate_dataset(total_steps: int):
    # 1. Pipeline Initialization
    print("Initializing Headless Raycast Simulator...")
    sim = RaycastSimulator(headless=True, layout='room_rectangular')
    
    print(f"Bypassing Cognitive Engine blocks... Starting execution for {total_steps} rapid steps.")
    
    start_time = time.time()
    reflex_queue = []
    
    # We start by drawing the initial frame to synchronize frame states before first action
    frame = sim.get_latest_frame()
    dist_str = sim.telemetry.get('dist', '0')
    sensor_dist = float(dist_str)
    
    chunk_size = 5000
    num_chunks = (total_steps + chunk_size - 1) // chunk_size
    steps_done = 0
    
    for chunk_idx in range(num_chunks):
        steps_this_chunk = min(chunk_size, total_steps - steps_done)
        
        # Ensure unique timestamp folder names if chunks complete faster than 1 second
        time.sleep(1.0)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = os.path.join(DATA_DIR, f'markov_simulated_{timestamp_str}')
        
        img_dir = os.path.join(base_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        meta_path = os.path.join(base_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({"controller": "MarkovSimulated"}, f, indent=4)
            
        csv_path = os.path.join(base_dir, 'episode_data.csv')
        
        # Open master telemetry for evaluation bounds
        master_tel_path = os.path.join(DATA_DIR, 'master_telemetry.csv')
        master_tel_exists = os.path.exists(master_tel_path)
        
        with open(csv_path, 'w', newline='') as f, open(master_tel_path, 'a', newline='') as mf:
            fieldnames = ['timestamp', 'image_file', 'ir_reading', 'batt_raw', 'ping_raw', 'pwm_left', 'pwm_right']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            m_fieldnames = ['img_dir', 'ts', 'cx', 'cy', 'dx', 'dy', 'dist_px', 'yaw_deg', 'ir']
            m_writer = csv.DictWriter(mf, fieldnames=m_fieldnames)
            if not master_tel_exists:
                m_writer.writeheader()
            
            print(f"\n--- Outputting Chunk {chunk_idx + 1}/{num_chunks} to {base_dir} ---")
            
            for i in range(steps_this_chunk):
                if i > 0 and i % 100 == 0:
                    elapsed = max(0.001, time.time() - start_time)
                    fps = steps_done / elapsed
                    print(f"{i}/{steps_this_chunk} images generated... ({fps:.2f} frames/sec)")

                # Trigger Reflex manually dynamically off raw math (matching AUTONOMY_THRESHOLD=1500 limit implicitly via 1200 bound logic)
                # A sensor_dist > 1500 triggers exactly around .33m dist.
                if sensor_dist > 1500 and not reflex_queue:
                    turn_dir = random.choice([3, 4])
                    reflex_queue.extend([2, turn_dir, turn_dir, turn_dir]) # Reversal -> Turn x3
                    
                # Act
                if reflex_queue:
                    action_id = reflex_queue.pop(0)
                else:
                    probs = [0.65, 0.05, 0.15, 0.15]
                    action_id = np.random.choice([1, 2, 3, 4], p=probs)
                    
                # Dispatch
                sim.send_command(action_id)
                
                # Fetch Next Frame & Data
                frame = sim.get_latest_frame()
                sim_tel = sim.telemetry
                sensor_dist = float(sim_tel.get('dist', '0'))
                
                # Fetch Math Ground Truth!
                x, y, theta = sim.pose
                yaw_deg = (math.degrees(theta) + 360) % 360
                
                # Build Timestamp
                ts_float = time.time()
                ts = str(ts_float)
                filename = f"frame_{int(ts_float * 1000)}.jpg"
                img_path = os.path.join(img_dir, filename)
                
                pwm_l, pwm_r = ACTION_PWM_MAP.get(action_id, (0, 0))
                
                row = {
                    'timestamp': ts,
                    'image_file': filename,
                    'ir_reading': str(int(sensor_dist)),
                    'batt_raw': '0',
                    'ping_raw': '0',
                    'pwm_left': str(pwm_l),
                    'pwm_right': str(pwm_r)
                }
                cv2.imwrite(img_path, frame)
                writer.writerow(row)
                
                # Export to master telemetry mapping (scaling 1m cell to ~200 bounding px for analytical parser parity)
                m_row = {
                    'img_dir': img_dir, 'ts': str(int(ts_float * 1000)),
                    'cx': float(x * 200.0), 'cy': float(y * 200.0),
                    'dx': float(math.cos(theta)), 'dy': float(math.sin(theta)),
                    'dist_px': 20.0, 'yaw_deg': float(yaw_deg), 'ir': int(sensor_dist)
                }
                m_writer.writerow(m_row)
                
                steps_done += 1

    print(f"\nGeneration Complete. {total_steps} frames saved successfully across {num_chunks} chunks.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulated markov transitions to generate")
    args = parser.parse_args()
    
    generate_dataset(args.steps)
