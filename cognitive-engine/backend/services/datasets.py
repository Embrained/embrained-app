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
import glob
import logging
import time
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from config import DATA_DIR, MODELS_DIR

logger = logging.getLogger("DatasetService")

class DatasetService:
    def __init__(self, data_root=None):
        self.data_root = data_root if data_root else DATA_DIR

    @staticmethod
    def get_video_frame(video_path, frame_idx):
        """Helper to extract a specific frame from a video file."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def load_transitions(self, dataset_path):
        """
        Standardized way to load transitions from a dataset folder.
        Handles 'markov' format (images + episode_data.csv).
        Returns a list of dicts: {'image_path' (or 'frame_idx'), 'action', 'next_image', ...}
        """
        has_csv = os.path.exists(os.path.join(dataset_path, "episode_data.csv"))
        transitions = []
        
        # [NEW] Handle Markov discrete bouts
        dataset_name = os.path.basename(dataset_path)
        if has_csv:
            csv_path = os.path.join(dataset_path, "episode_data.csv")
            img_dir = os.path.join(dataset_path, "images")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for i in range(len(df)):
                    row = df.iloc[i]
                    img_file = str(row.get('image_file', f'frame_{i}.jpg'))
                    img_path = os.path.join(img_dir, img_file)
                    
                    def fval(key, default=0.0):
                        val = row.get(key, default)
                        try:
                            return float(val) if not pd.isna(val) else default
                        except Exception:
                            return default
                            
                    l_val = fval('pwm_left', 0.0)
                    r_val = fval('pwm_right', 0.0)
                    
                    # Compute nearest macro_action ID mathcing ACTION_PWM_MAP config
                    from config import ACTION_PWM_MAP
                    import math
                    best_action = 0
                    best_dist = float('inf')
                    for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
                        dist = math.hypot(l_val - map_l, r_val - map_r)
                        if dist < best_dist:
                            best_dist = dist
                            best_action = act_id
                            
                    transitions.append({
                        'format': 'markov',
                        'image_path': img_path,
                        'frame_idx': i,
                        'timestamp': fval('timestamp', time.time()),
                        'action': [l_val, r_val],
                        'state': [l_val, r_val],
                        'dist': fval('ir_reading', 0.0),
                        'ping': 0.0,
                        'macro_action': best_action,
                        'left_cmd': l_val,
                        'right_cmd': r_val,
                        'session': dataset_name
                    })
                    
        return transitions

    def list_datasets(self, path=None, fast=False):
        """List all dataset directories in the specified path or ./data default."""
        data_dir = os.path.abspath(self.data_root)

        # Force absolute and normalized
        if path and path.strip():
            # [FIX] Handle potential mixed slashes from frontend
            path_str = str(path).replace('\\', '/')
            
            # [FIX] Cross-OS Path Sanitization
            if len(path_str) > 2 and path_str[1] == ':' and path_str[2] == '/':
                path_str = path_str[2:] # Strip Windows drive letter
                
            data_root_fwd = str(self.data_root).replace('\\', '/')
            if os.path.isabs(path_str) and not path_str.startswith(data_root_fwd):
                parts = path_str.split('/')
                if "data" in parts:
                    idx = parts.index("data")
                    subpath = "/".join(parts[idx+1:])
                    path_str = subpath if subpath else ""
                else:
                    path_str = os.path.basename(path_str)

            if os.path.isabs(path_str):
                potential_dir = os.path.abspath(path_str)
            else:
                potential_dir = os.path.abspath(os.path.join(self.data_root, path_str))
                
            if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                data_dir = potential_dir
                
        data_dir = os.path.normpath(data_dir)

        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
                # Create a placeholder file as requested
                with open(os.path.join(data_dir, "placeholder.txt"), "w") as f:
                    f.write("Datasets will be stored here.\n")
                logger.info(f"Created missing dataset directory: {data_dir}")
            except Exception as e:
                logger.warning(f"Failed to create dataset directory {data_dir}: {e}")
            return {"datasets": [], "root": data_dir}
        
        if not os.path.isdir(data_dir):
            logger.warning(f"list_datasets: Path is not a directory: {data_dir}")
            print("LIST_DATASETS NOT DIR:", data_dir)
            return {"datasets": [], "root": data_dir}

        print("LIST_DATASETS SUCCESS DIR:", data_dir)

        datasets_map = {}
        try:
            items = os.listdir(data_dir)
            for d in items:
                full_path = os.path.join(data_dir, d)
                if os.path.isdir(full_path) and d != "logs":
                    # Check for markov format
                    has_csv = os.path.exists(os.path.join(full_path, "episode_data.csv"))
                    has_markov = has_csv
                    
                    if has_markov:
                        import re
                        # Format is typically '{prefix}_{YYYY}-{MM}-{DD}_{HH}-{MM}-{SS}'
                        # We extract '{prefix}'
                        match = re.match(r'^(.*)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$', d)
                        if match:
                            prefix = match.group(1)
                        else:
                            prefix = d
                            
                        count = self._get_dataset_length(full_path)
                        
                        if prefix not in datasets_map:
                            datasets_map[prefix] = {
                                "name": prefix,
                                "count": 0,
                                "format": "markov"
                            }
                        if count > 0:
                            datasets_map[prefix]["count"] += count
        except Exception as e:
            logger.error(f"Error listing datasets in {data_dir}: {e}")

        datasets = list(datasets_map.values())
        datasets.sort(key=lambda x: x["name"])
        print("LIST_DATASETS RETURNING:", len(datasets))
        return {"datasets": datasets, "root": data_dir}

    def _get_dataset_length(self, dataset_path):
        try:
            csv_path = os.path.join(dataset_path, "episode_data.csv")
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    return max(0, sum(1 for line in f) - 1)
        except Exception:
            pass
        return 0

    def get_dataset_count(self, name, path=None):
        data_dir = path if path else self.data_root
        full_path = os.path.join(data_dir, name)
        return self._get_dataset_length(full_path)

    def list_images(self, path):
        """List images in a specific data subdirectory."""
        data_dir = self.data_root
        target_dir = os.path.join(data_dir, path)
        
        if not os.path.exists(target_dir):
            if os.path.exists(path):
                target_dir = path
            else:
                return []

        images = []
        search_dirs = [target_dir, os.path.join(target_dir, "images")]
        
        for d in search_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                         full_path = os.path.join(d, f)
                         try:
                             rel_path = os.path.relpath(full_path, start=os.getcwd())
                         except:
                             rel_path = full_path
                         
                         images.append({
                             "name": f,
                             "path": rel_path,
                             "full_path": full_path
                         })
        images.sort(key=lambda x: x['name'])
        return images

    def list_models(self, path=None):
        """Lists available trained models (Policies and VAEs)."""
        search_dirs = [MODELS_DIR, self.data_root]
        if path and os.path.exists(path):
            search_dirs = [path, MODELS_DIR]

        # 1. Policies
        cql_files = []
        for d in search_dirs:
            if path:
                 cql_files.extend(glob.glob(os.path.join(d, "*cql*.pth")))
                 cql_files.extend(glob.glob(os.path.join(d, "*", "*cql*.pth")))
                 cql_files.extend(glob.glob(os.path.join(d, "*_policy.pth")))
                 cql_files.extend(glob.glob(os.path.join(d, "*", "*_policy.pth")))
                 cql_files.extend(glob.glob(os.path.join(d, "*fixed_goal*.pth")))
                 cql_files.extend(glob.glob(os.path.join(d, "*hello_world*.pth")))
                 cql_files.extend(glob.glob(os.path.join(d, "*e2e*.pth")))
                 cql_files.extend(glob.glob(os.path.join(d, "*", "*e2e*.pth")))
            else:
                 cql_files.extend(glob.glob(os.path.join(d, "**", "*cql*.pth"), recursive=True))
                 cql_files.extend(glob.glob(os.path.join(d, "**", "*_policy.pth"), recursive=True))
                 cql_files.extend(glob.glob(os.path.join(d, "**", "*fixed_goal*.pth"), recursive=True))
                 cql_files.extend(glob.glob(os.path.join(d, "**", "*hello_world*.pth"), recursive=True))
                 cql_files.extend(glob.glob(os.path.join(d, "**", "*e2e*.pth"), recursive=True))

        # 2. VAEs
        vae_files = []
        for d in search_dirs:
            all_vaes = []
            if path:
                 all_vaes.extend(glob.glob(os.path.join(d, "*vae*.pth"))) 
                 all_vaes.extend(glob.glob(os.path.join(d, "*", "*vae*.pth"))) 
            else:
                 vae_files.extend(glob.glob(os.path.join(d, "**", "*tiny_vae_final.pth"), recursive=True))
                 vae_files.extend(glob.glob(os.path.join(d, "**", "*_vae_encoder.pth"), recursive=True))
                 all_vaes.extend(glob.glob(os.path.join(d, "**", "*vae*.pth"), recursive=True))
                 
            for v in all_vaes:
                base = os.path.basename(v).lower()
                if not any(x in base for x in ["cql", "latentslam", "control", "model", "policy"]):
                    vae_files.append(v)
        
        # 3. SLAM Models
        slam_files = []
        for d in search_dirs:
            if path:
                 slam_files.extend(glob.glob(os.path.join(d, "*latentslam*.pth")))
                 slam_files.extend(glob.glob(os.path.join(d, "*", "*latentslam*.pth")))
            else:
                 slam_files.extend(glob.glob(os.path.join(d, "**", "*latentslam*.pth"), recursive=True))

        # Filter out intermediate training checkpoints
        slam_files = [f for f in slam_files if "latentslam_best.pth" not in os.path.basename(f)]

        final_policies = self._process_file_list(cql_files, is_policy=True)
        final_vaes = self._process_file_list(vae_files, is_policy=False)
        final_slam = self._process_file_list(slam_files, is_policy=False)
        
        # Add exploration heuristics to controllers list
        heuristic_algos = [
            {"name": "Markov", "path": "", "modified": time.time(), "modified_str": "", "size_mb": 0.0},
            {"name": "MarkovWASD", "path": "", "modified": time.time(), "modified_str": "", "size_mb": 0.0},
            {"name": "MarkovSweep", "path": "", "modified": time.time(), "modified_str": "", "size_mb": 0.0},
            {"name": "Algorithmic Oracle", "path": "", "modified": time.time(), "modified_str": "", "size_mb": 0.0}
        ]
        
        final_policies.extend(heuristic_algos)
        
        return {"models": final_policies, "vaes": final_vaes, "slam": []}

    def _process_file_list(self, files, is_policy=False):
        processed = []
        seen = set()
        
        for p in files:
            name = os.path.basename(p)
            if name in seen: continue
            
            try:
                mtime = os.path.getmtime(p)
                size = os.path.getsize(p)
                info = {
                    "name": name,
                    "path": p,
                    "modified": mtime,
                    "modified_str": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)),
                    "size_mb": round(size / (1024*1024), 2)
                }
                
                if is_policy:
                    # Policy specific metadata (goals)
                    parent = os.path.dirname(p)
                    possible_dataset_path = parent if parent != MODELS_DIR else None
                    if not possible_dataset_path:
                         parts = name.split('-vae')
                         if len(parts) > 0:
                             possible_dataset_path = os.path.join(self.data_root, parts[0])
                    
                    if possible_dataset_path and os.path.exists(possible_dataset_path):
                         base_policy_name = os.path.basename(p)
                         if "-cql" in base_policy_name:
                             vae_name = base_policy_name.split("-cql")[0]
                         else:
                             vae_name = base_policy_name.replace("_policy.pth", "")
                             
                         goals_dir = os.path.join(possible_dataset_path, f"{vae_name}_goals")
                         if os.path.exists(goals_dir):
                             imgs = glob.glob(os.path.join(goals_dir, "*.jpg")) + glob.glob(os.path.join(goals_dir, "*.png"))
                             info['goal_count'] = len(imgs)
                             info['dataset'] = os.path.basename(possible_dataset_path)

                processed.append(info)
                seen.add(name)
            except Exception:
                pass
                
        processed.sort(key=lambda x: x['modified'], reverse=True)
        return processed
