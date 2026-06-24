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
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy
import random
import cv2
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys
import time
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HIDDEN_DIM, INPUT_DIM, MODELS_DIR, ACTION_DIM, ACTION_PWM_MAP

# Import new Spatial Models
from modules.spatial_model import TinyVAE, CQLNetwork, FullQNet

# Configure Logging
logger = logging.getLogger("TrainSpatialCQL")
logging.basicConfig(level=logging.INFO)

# --- Hyperparameters ---
# BATCH_SIZE and LEARNING_RATE are now passed dynamically
GAMMA = 0.90 # [UPDATED] Discount factor tuned for 10-step horizon
COSINE_ALPHA = 1.0 # Reward scaling
NUM_EPOCHS = 50
# HIDDEN_DIM and INPUT_DIM imported from config
MAX_HER_HORIZON = 10 # [RESTORED] Horizon expanded to bridge rotational sweeps

IMG_H = 64
IMG_W = 64

# Transform for TinyVAE (Resize only, no normalize usually if sigmoid output, but we use tensor [0,1])
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    # T.Normalize ... VAE usually expects [0, 1] range if Sigmoid output. Unnormalized is fine.
])

class SpatialCQLDataset(Dataset):
    def __init__(self, episodes, data_root, device, progress_callback=None, dataset_percent=100, goal_type=None, image_size=64, codebook=None):
        self.episodes = episodes
        self.data_root = data_root
        self.device = device
        self.codebook = codebook
        self.goal_type = goal_type # [NEW] Store goal_type for reward logic
        self.image_size = image_size
        self.samples = []
        self.valid_actions = set() # [NEW] Track actions seen in data
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])
        
        self.image_cache = {}
        
        logger.debug("SpatialCQLDataset: Verifying data integrity (checking file paths)...")
        
        missing_count = 0
        total_eps = len(episodes)
        
        # [NEW] Action Counter for Balancing
        action_counts = {} 
        
        for idx, traj in enumerate(episodes):
            if idx % 50 == 0:
                logger.debug(f"Processing Trajectory {idx}/{total_eps} ... ({len(self.samples)} samples collected)")
            if progress_callback and idx % 10 == 0:
                progress_callback(0, float(idx)/total_eps) # Epoch 0 for setup phase
                
            full_seq = traj # Input is now the full sequence directly
            
            seq_len = len(full_seq)
            
            for start_idx in range(seq_len - 1):
                # Inner loop: potential future goals
                # We start from start_idx + 1 (next step) up to end
                for goal_idx in range(start_idx + 1, seq_len):
                    
                    # 1. Local Efficiency Filter
                    temporal_dist = goal_idx - start_idx
                    if temporal_dist > MAX_HER_HORIZON:
                        continue # Skip long random walks
                        
                    # [FIX] If the goal type does not utilize HER goal contexts, 
                    # we only need ONE copy of the transition, not MAX_HER_HORIZON copies!
                    if self.goal_type in ['dark_wall_seek', 'group_goal', 'discrete_exact']:
                        if goal_idx > start_idx + 1:
                            continue
                        
                    curr_node = full_seq[start_idx]
                    
                    # [UPDATED] Direct State Transition (SMDP)
                    # Next state is simply the immediate next frame in the sequence
                    next_idx = start_idx + 1
                    if next_idx >= seq_len:
                        continue 
                        
                    next_node = full_seq[next_idx]
                    
                    goal_node = full_seq[goal_idx]
                         
                    # 2. Add Standard Transition (Hindsight Relabeled)
                    # "To go from start_idx to goal_idx, taking derived PWM action is valid."
                    
                    # Extract macro action (WASD ID 0-4) from dataset
                    if 'macro_action' in curr_node:
                        action = int(curr_node['macro_action'])
                        if action >= ACTION_DIM:
                            action = 0 # Fallback for safety
                    else:
                        # Fallback to closest 5-class bin if macro_action is missing
                        raw_l = float(curr_node.get('left_cmd', 0.0))
                        raw_r = float(curr_node.get('right_cmd', 0.0))
                        
                        best_action = 0
                        best_dist = float('inf')
                        for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
                            dist = math.hypot(raw_l - map_l, raw_r - map_r)
                            if dist < best_dist:
                                best_dist = dist
                                best_action = act_id
                        action = best_action
                    
                    # 3. Filter to only FWD and TURN and REVERSE actions (1=FWD, 2=REVERSE, 3=LEFT, 4=RIGHT)
                    if action not in [1, 2, 3, 4]:
                        continue
                        
                    # Track actions
                    self.valid_actions.add(action)
                    action_counts[action] = action_counts.get(action, 0) + 1
                    
                    # HER Reward Logic
                    if self.goal_type == 'ir_wall_seeking':
                        # Wall seeking: Reward based on IR reading > 200
                        ir_reading = float(next_node.get('dist', next_node.get('sonar', 0.0)))
                        is_local_done = ir_reading > 200.0
                        reward = 1.0 if is_local_done else -0.01
                    elif self.goal_type == 'dark_wall_seek':
                        # Pure reflex dark wall seeking based on image brightness (top 50% only)
                        if 'luminance_reward' not in next_node:
                            img_tensor = self._load_img(next_node)
                            if img_tensor is not None and img_tensor.shape[0] == 3:
                                luminance = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
                                h = luminance.shape[0]
                                mean_brightness = luminance[:h//2, :].mean().item()
                                if mean_brightness < 0.35:
                                    next_node['luminance_reward'] = 1.0
                                else:
                                    next_node['luminance_reward'] = -0.5
                            else:
                                next_node['luminance_reward'] = -0.5
                        reward = next_node['luminance_reward']
                        is_local_done = False
                    elif self.goal_type == 'group_goal' or self.goal_type == 'discrete_exact':
                        # Placeholder, dynamically evaluated in __getitem__ using precalculated latents
                        reward = 0.0
                        is_local_done = False
                    else:
                        # If next_node leaps to or past the goal state, it has arrived.
                        is_local_done = (next_idx >= goal_idx)
                        
                        if is_local_done:
                            reward = 1.0
                        else:
                            reward = -0.01 # Step penalty
                             
                    # Simplify to 1-step (current frame)
                    def get_historical_action(idx):
                        if idx <= 0: return 0  # STOP if no history
                        prev_node = full_seq[idx - 1]
                        if 'macro_action' in prev_node:
                            act = int(prev_node['macro_action'])
                            return act if act < ACTION_DIM else 0
                        else:
                            raw_l = float(prev_node.get('left_cmd', 0.0))
                            raw_r = float(prev_node.get('right_cmd', 0.0))
                            best_action = 0
                            best_dist = float('inf')
                            for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
                                d = math.hypot(raw_l - map_l, raw_r - map_r)
                                if d < best_dist:
                                    best_dist = d
                                    best_action = act_id
                            return best_action

                    def get_node(idx):
                        if idx < 0: return full_seq[0]
                        if idx >= seq_len: return full_seq[seq_len - 1]
                        return full_seq[idx]
                        
                    # [REMOVED] Historical Frame Vector Stacking
                    # The environment is perfectly Markovian (complete stops), so historical
                    # velocity is not required. State consists of a single frame.
                    if random.random() < 0.30:
                        curr_nodes_stack = [get_node(start_idx)]
                        next_nodes_stack = [get_node(next_idx)]
                        curr_actions_stack = [get_historical_action(start_idx)]
                        next_actions_stack = [get_historical_action(next_idx)]
                    else:
                        curr_nodes_stack = [get_node(start_idx)]
                        next_nodes_stack = [get_node(next_idx)]
                        curr_actions_stack = [get_historical_action(start_idx)]
                        next_actions_stack = [get_historical_action(next_idx)]
                    
                    # [NEW] Enforce pure reflex isolation for dark_wall_seek
                    effective_goal_node = None if self.goal_type in ['dark_wall_seek', 'group_goal', 'discrete_exact'] else goal_node
                    
                    self.samples.append({
                        'curr_nodes': curr_nodes_stack,
                        'next_nodes': next_nodes_stack,
                        'curr_actions': curr_actions_stack,
                        'next_actions': next_actions_stack,
                        'goal_node': effective_goal_node,
                        'action': action,
                        'reward': reward,
                        'done': is_local_done # Task is locally done
                    })

        if missing_count > 0:
            logger.warning(f"Skipped {missing_count} transitions due to missing files (or bad paths).")
            
        logger.debug(f"Dataset unique actions: {sorted(list(self.valid_actions))}")
        logger.debug(f"Action Counts: {action_counts}")

        # No more manual STOP downsampling; relying on WeightedRandomSampler to balance 5 classes.
        random.shuffle(self.samples)
        
        # Re-count for logging
        final_counts = {}
        for s in self.samples:
            act = s['action']
            final_counts[act] = final_counts.get(act, 0) + 1
            
        logger.debug(f"Dataset Size: {len(self.samples)}")
        logger.debug(f"Class Counts: {final_counts}")
        
        # [NEW] Percentage Subsampling (Speed Optimization)
        # Apply AFTER balancing to preserve ratio
        logger.debug(f"Requested Dataset Percent: {dataset_percent}%")
        if dataset_percent < 100:
             target_size = int(len(self.samples) * (dataset_percent / 100.0))
             target_size = max(target_size, 100) # Safety floor
             logger.debug(f"Subsampling to {dataset_percent}%: Reducing {len(self.samples)} -> {target_size}")
             
             random.shuffle(self.samples)
             self.samples = self.samples[:target_size]
                
        logger.debug(f"Using Balanced Dataset: {len(self.samples)} samples.")
        
        # [NEW] Calculate Sample Weights
        # 1. Recalculate counts based on FINAL dataset (post-downsampling)
        final_action_counts = {}
        for s in self.samples:
            act = s['action']
            final_action_counts[act] = final_action_counts.get(act, 0) + 1
            
        logger.debug(f"Final Action Counts for Weighting: {final_action_counts}")

        # 2. Inverse frequency based on FINAL counts
        class_weights = {}
        for act, count in final_action_counts.items():
            if count > 0:
                class_weights[act] = 1.0 / count
        
        # 2. Assign weight to each sample
        self.sample_weights = []
        for s in self.samples:
            w = class_weights.get(s['action'], 0.0)
            self.sample_weights.append(w)
            
        # [NEW] Pre-populate entire frame RAM cache to instantly bypass I/O
        # (DrQ typically executes millions of read() operations if not cached)
        unique_paths = set()
        for sample in self.samples:
            nodes_to_cache = sample['curr_nodes'] + sample['next_nodes']
            if sample.get('goal_node'):
                nodes_to_cache.append(sample['goal_node'])
                
            for n in nodes_to_cache:
                if n and 'image_path' in n:
                    p = n['image_path']
                    if not os.path.isabs(p):
                        p = os.path.join(self.data_root, p)
                    unique_paths.add(p)
                    
        logger.info(f"Caching {len(unique_paths)} distinct frames into RAM for blazing fast I/O...")
        fail_count = 0
        for p in unique_paths:
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    self.image_cache[p] = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    fail_count += 1
            else:
                fail_count += 1
                
        if fail_count > 0:
            logger.warning(f"Note: {fail_count} frames could not be read during caching.")
            
        self.valid_actions.add(5) # [NEW] Force STOP action (5) to be considered a valid action so the model can output it at the goal
    
    def _load_img(self, node):
        if not node:
            return torch.zeros((3, self.image_size, self.image_size))
            
        try:
            # 1. Check for explicit image path (Markov sequence)
            if 'image_path' in node:
                p = node['image_path']
                if not os.path.isabs(p):
                    p = os.path.join(self.data_root, p)
                
                if p in self.image_cache:
                    return self.image_cache[p]
                
                if os.path.exists(p):
                            
                    # Ignore webcam, directly load egocentric frame
                    img = cv2.imread(p)
                    if img is not None:
                        t = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        self.image_cache[p] = t
                        return t
            else:
                 logger.error(f"Node missing image_path. Only Markov formatted data is supported.")

        except Exception as e:
            logger.error(f"Error loading image for node: {e}")
            
        return torch.zeros((3, self.image_size, self.image_size))

    def _extract_state(self, node, action_id=0):
        """
        [MODIFIED] Returns a 0-Dimensional PyTorch tensor natively to drop the telemetry logic permanently 
        while preserving the DataLoader's iterative structure downstream. 
        """
        return torch.tensor([], dtype=torch.float)

    def precompute_latents(self, encoder, device):
        unique_nodes = []
        node_ids = set()
        
        for sample in self.samples:
            nodes_to_process = sample['curr_nodes'] + sample['next_nodes']
            if sample.get('goal_node'):
                nodes_to_process.append(sample['goal_node'])
                
            for n in nodes_to_process:
                if n and id(n) not in node_ids:
                    node_ids.add(id(n))
                    unique_nodes.append(n)
                    
        logger.info(f"Pre-computing VAE latents for {len(unique_nodes)} unique frames...")
        encoder.eval()
        
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(unique_nodes), batch_size):
                batch_nodes = unique_nodes[i:i+batch_size]
                imgs = torch.stack([self._load_img(n) for n in batch_nodes]).to(device)
                if hasattr(encoder, 'vq'):
                    _, _, mus, _, _ = encoder(imgs)
                else:
                    feats = encoder.encoder(imgs)
                    mus = encoder.fc_mu(feats)
                for j, n in enumerate(batch_nodes):
                    n['latent'] = mus[j].cpu()
                    
        self.use_precomputed = True
        
        # [NEW] Precompute Group Goal Stats
        if self.goal_type == 'group_goal':
            import json
            stats_path = os.path.join(self.data_root, 'goals', 'group_stats.json')
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    self.group_centroid = torch.tensor(stats['centroid'], device=device, dtype=torch.float)
                    self.group_avg_dist = stats['average_in_group_distance']
                    logger.info(f"Loaded Group Goal centroid. In-group avg dist: {self.group_avg_dist:.4f}")
                except Exception as e:
                    logger.error(f"Failed to load group_stats.json: {e}")
                    self.group_centroid = None
            else:
                logger.error("group_stats.json not found. Group Goal logic will fail!")
                self.group_centroid = None
                
        # [NEW] Precompute Discrete Exact Stats
        if self.goal_type == 'discrete_exact':
            import json
            stats_path = os.path.join(self.data_root, 'goals', 'discrete_exact_stats.json')
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    self.exact_latent = torch.tensor(stats['exact_latent'], device=device, dtype=torch.float)
                    if 'exact_latents' in stats:
                        self.exact_latents = torch.tensor(stats['exact_latents'], device=device, dtype=torch.float)
                    else:
                        self.exact_latents = self.exact_latent.unsqueeze(0)
                    logger.info(f"Loaded discrete exact goal latents (Count: {self.exact_latents.shape[0]}).")
                except Exception as e:
                    logger.error(f"Failed to load discrete_exact_stats.json: {e}")
                    self.exact_latent = None
            else:
                logger.error("discrete_exact_stats.json not found. Discrete Exact logic will fail!")
                self.exact_latent = None

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if getattr(self, 'use_precomputed', False):
            latent_curr_stack = torch.stack([n.get('latent', torch.zeros(32)) for n in sample['curr_nodes']], dim=0)
            latent_next_stack = torch.stack([n.get('latent', torch.zeros(32)) for n in sample['next_nodes']], dim=0)
            latent_goal = sample['goal_node'].get('latent', torch.zeros(32)) if sample.get('goal_node') else torch.zeros(32)
        else:
            # Load raw images instead of pre-computed latents (for DrQ CNN End-To-End)
            latent_curr_stack = torch.stack([self._load_img(n) for n in sample['curr_nodes']], dim=0)
            latent_next_stack = torch.stack([self._load_img(n) for n in sample['next_nodes']], dim=0)
            latent_goal = self._load_img(sample['goal_node'])
        
        # Frame stacked explicit state
        state_curr_stack = torch.cat([self._extract_state(n, action_id=a) for n, a in zip(sample['curr_nodes'], sample['curr_actions'])], dim=0)
        state_next_stack = torch.cat([self._extract_state(n, action_id=a) for n, a in zip(sample['next_nodes'], sample['next_actions'])], dim=0)

        # Dynamic Override for Group Goal
        final_reward = torch.tensor(sample['reward'], dtype=torch.float)
        final_done = torch.tensor(sample['done'], dtype=torch.float)
        
        final_action = sample['action']
        if getattr(self, 'goal_type', None) == 'group_goal' and getattr(self, 'group_centroid', None) is not None:
            # Distance of the most recent latent frame to the group centroid
            latest_latent = latent_next_stack[-1] # Usually stacked 3 frames, last one is current
            
            # Slice to match the centroid's dimension (e.g. 32), ignoring any injected trajectory telemetry
            vision_dim = self.group_centroid.shape[-1]
            vision_latest_latent = latest_latent[:vision_dim]
            
            if vision_latest_latent.dim() == 1:
                dist = torch.norm(vision_latest_latent.to(self.group_centroid.device) - self.group_centroid)
            else:
                dist = torch.norm(vision_latest_latent.view(-1).to(self.group_centroid.device) - self.group_centroid)
                
            if dist < (self.group_avg_dist * 1.1):
                final_reward = torch.tensor(50.0, dtype=torch.float)
                final_done = torch.tensor(1.0, dtype=torch.float)
                final_action = 5 # Force STOP action at goal
            else:
                final_reward = torch.tensor(-0.01, dtype=torch.float)
                final_done = torch.tensor(0.0, dtype=torch.float)

        if getattr(self, 'goal_type', None) == 'discrete_exact' and getattr(self, 'exact_latent', None) is not None:
            # For discrete_exact, terminal assignment is now strictly controlled by the HER phase tagging
            # to support downsampling without conflicting Q-values.
            if getattr(self, 'terminal_indices_set', None) is not None:
                if idx in self.terminal_indices_set:
                    final_reward = torch.tensor(1.0, dtype=torch.float)
                    final_done = torch.tensor(1.0, dtype=torch.float)
                    final_action = 5
                else:
                    if final_action == 5:
                        final_reward = torch.tensor(-1.0, dtype=torch.float)
                        final_done = torch.tensor(1.0, dtype=torch.float)
                    else:
                        final_reward = torch.tensor(0.0, dtype=torch.float)
                        final_done = torch.tensor(0.0, dtype=torch.float)
            else:
                # Fallback before HER tagging
                latest_latent = latent_next_stack[-1]
                vision_dim = self.exact_latent.shape[-1]
                vision_latest_latent = latest_latent[:vision_dim].view(-1).to(self.exact_latents.device)
                
                dists = torch.norm(self.exact_latents - vision_latest_latent.unsqueeze(0), dim=1)
                    
                if torch.any(dists < 0.001):
                    final_reward = torch.tensor(1.0, dtype=torch.float)
                    final_done = torch.tensor(1.0, dtype=torch.float)
                    final_action = 5
                else:
                    final_reward = torch.tensor(0.0, dtype=torch.float)
                    final_done = torch.tensor(0.0, dtype=torch.float)

        if getattr(self, 'goal_type', None) == 'discrete_exact':
            if getattr(self, 'codebook', None) is not None:
                def _to_onehot(latent_stack):
                    if latent_stack.dim() == 1:
                        latent_stack = latent_stack.unsqueeze(0)
                        squeeze = True
                    else:
                        squeeze = False
                        
                    d = torch.sum(latent_stack ** 2, dim=-1, keepdim=True) + \
                        torch.sum(self.codebook ** 2, dim=1) - \
                        2 * torch.matmul(latent_stack, self.codebook.t())
                    idx = torch.argmin(d, dim=-1)
                    oh = F.one_hot(idx, num_classes=self.codebook.shape[0]).float()
                    if squeeze: oh = oh.squeeze(0)
                    return oh
                
                latent_curr_stack = _to_onehot(latent_curr_stack.to(self.codebook.device)).cpu()
                latent_next_stack = _to_onehot(latent_next_stack.to(self.codebook.device)).cpu()
                latent_goal = _to_onehot(latent_goal.to(self.codebook.device)).cpu()

        return (
            latent_curr_stack, 
            latent_next_stack, 
            latent_goal, 
            state_curr_stack,
            state_next_stack,
            torch.tensor(final_action, dtype=torch.long), 
            final_reward, 
            final_done
        )

def train(data_root, num_epochs=20, stop_event=None, progress_callback=None, vae_model_filename=None, batch_size=128, learning_rate=1e-5, model_size='large', dataset_percent=10, goal_type='her', alpha=0.1, selected_datasets=None, model_filename=None, train_from_scratch=False, use_telemetry=False):
    from torch.utils.data import WeightedRandomSampler # [NEW]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"Training on {device} (CUDA Available: {torch.cuda.is_available()})")
    logger.debug(f"Hyperparameters: Batch Size={batch_size}, LR={learning_rate}, Size={model_size}")

    # --- [NEW] Resolve VAE Path & Infers Data Root ---
    vae_path = None
    if vae_model_filename:
        # 1. Try absolute or direct path provided
        if os.path.exists(vae_model_filename):
             vae_path = vae_model_filename
        # 2. Try in models dir
        elif os.path.exists(os.path.join(MODELS_DIR, vae_model_filename)):
             vae_path = os.path.join(MODELS_DIR, vae_model_filename)
        # 3. Try in data dir
        elif os.path.exists(os.path.join(data_root, vae_model_filename)):
             vae_path = os.path.join(data_root, vae_model_filename)
        # 4. Recursive Search in Data Dir (to match frontend listing)
        else:
             import glob
             candidates = glob.glob(os.path.join(data_root, "**", vae_model_filename), recursive=True)
             if candidates:
                 vae_path = candidates[0]

    # Load Transitions instead of Episodes
    trans_path = os.path.join(data_root, "all_transitions.json")
    if not os.path.exists(trans_path):
        # Fallback logic if needed, or error
        if vae_path:
             # Try VAE dir
             vae_dir = os.path.dirname(vae_path)
             alt_trans = os.path.join(vae_dir, "all_transitions.json")
             if os.path.exists(alt_trans):
                 data_root = vae_dir
                 trans_path = alt_trans
        
        if not os.path.exists(trans_path):
             raise FileNotFoundError(f"all_transitions.json not found in {data_root}.")

    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    # Group by Session
    sessions = {}
    for item in all_data:
        s = item['session']
        if s not in sessions: sessions[s] = []
        sessions[s].append(item)
        
    # Create Trajectories
    trajectories = []
    logger.debug(f"Found {len(sessions)} sessions. Constructing trajectories...")
    for s in sessions:
        # Sort by timestamp
        traj = sorted(sessions[s], key=lambda x: x['timestamp'])
        if len(traj) > 5: # Filter tiny non-useful segments
            trajectories.append(traj)
            
    logger.debug(f"Generated {len(trajectories)} valid trajectories from raw transitions.")
        
    loaded_state = None
    model_size_detected = 'small' # Default
    latent_dim_detected = 32      # Default
    input_spatial_dim_detected = 64 # Default
    in_channels_detected = 3
    encoder = None # Initialize empty encoder for groundtruth skip

    is_groundtruth = vae_model_filename == "master_telemetry.csv"

    if is_groundtruth:
        logger.info("Ground Truth Mode Activated. Loading Allocentric Physics Vectors natively.")
        latent_dim_detected = 4
        
        telemetry_dict = {}
        telemetry_path = vae_path if vae_path and os.path.exists(vae_path) else os.path.join(data_root, "master_telemetry.csv")
        if os.path.exists(telemetry_path):
            import pandas as pd
            import math
            df = pd.read_csv(telemetry_path)
            for _, row in df.iterrows():
                yaw_rad = math.radians(row['yaw_deg'])
                telemetry_dict[str(row['ts'])] = torch.tensor([
                    row['cx'] / 640.0,
                    row['cy'] / 480.0,
                    math.cos(yaw_rad),
                    math.sin(yaw_rad)
                ], dtype=torch.float)
        else:
            raise FileNotFoundError(f"Missing master_telemetry.csv in {data_root}")

        unique_nodes_count = 0
        for traj_idx, traj in enumerate(trajectories):
            for node in traj:
                if 'latent' not in node:
                    ts = str(node.get('timestamp', ''))
                    
                    if ts == '':
                        # Fallback parsing if timestamp missing cleanly
                        path = node.get('image_path', '')
                        if path:
                            base = os.path.basename(path)
                            ts = base.replace('frame_', '').replace('webcam_frame_', '').replace('.jpg', '')
                            
                    if ts in telemetry_dict:
                         node['latent'] = telemetry_dict[ts]
                         unique_nodes_count += 1
                    else:
                         node['latent'] = torch.zeros(4, dtype=torch.float)
                         
        logger.debug(f"Successfully cached {unique_nodes_count} physics latents in memory.")
    else:
        # Load Pretrained VAE Weights
        # vae_path is already partially resolved above, but check if we need to auto-discover
        if not vae_path and not train_from_scratch:
            # Fallback to auto-discovery if explicit name failed or wasn't provided
            parent_name = os.path.basename(os.path.normpath(data_root))
            if not parent_name or parent_name == 'data' or parent_name == '.':
                prefix = ""
            else:
                prefix = f"{parent_name}_"
                
            possible_new = os.path.join(MODELS_DIR, f"{parent_name}-vae.pth")
            if os.path.exists(possible_new):
                 vae_path = possible_new
            else:
                 raise FileNotFoundError(f"Could not find VAE model: {possible_new}. Please train the VAE for this dataset first.")
                 
        if vae_path and os.path.exists(vae_path):
            try:
                temp_state = torch.load(vae_path, map_location=device, weights_only=True)
                
                if "valid_actions" in temp_state or "model_state_dict" in temp_state:
                    logger.warning(f"The file {vae_path} appears to be a CQL Policy, not a VAE Encoder!")
                    alt_path = vae_path.replace("-cql", "").replace("cql_", "")
                    if os.path.exists(alt_path) and alt_path != vae_path:
                        vae_path = alt_path
                        temp_state = torch.load(vae_path, map_location=device, weights_only=True)
                        if "valid_actions" in temp_state or "model_state_dict" in temp_state:
                             raise ValueError("Auto-corrected file is ALSO a policy.")
                    else:
                        parent = os.path.dirname(vae_path)
                        candidates = [f for f in os.listdir(parent) if "vae" in f.lower() and "cql" not in f.lower() and f.endswith(".pth")]
                        if candidates:
                            vae_path = os.path.join(parent, candidates[0])
                            temp_state = torch.load(vae_path, map_location=device, weights_only=True)
                        else:
                            raise ValueError(f"No valid VAE found to replace {vae_path}")

                loaded_state = temp_state
                
                latent_dim_detected, model_size_detected, input_spatial_dim_detected, in_channels_detected = TinyVAE.detect_size(loaded_state)
                logger.debug(f"Detected VAE: Size={model_size_detected.upper()}, Latent={latent_dim_detected}, Spatial={input_spatial_dim_detected}, Channels={in_channels_detected}")
                
            except Exception as e:
                logger.error(f"Failed to inspect VAE at {vae_path}: {e}")
                raise e
        else:
            logger.warning(f"No VAE weights found at {vae_path}. Encoder will be random (Bad!)")

        # Dynamically override the dataset image transforms 
        global transform, IMG_H, IMG_W
        if input_spatial_dim_detected != IMG_W:
            IMG_H = input_spatial_dim_detected
            IMG_W = input_spatial_dim_detected
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((IMG_H, IMG_W)),
                T.ToTensor(),
            ])
            logger.info(f"Dynamically resized offline image transform to {IMG_W}x{IMG_H} to match intrinsic VAE requirements.")

        # Init Models with CORRECT SIZE
        in_channels_detected = in_channels_detected if 'in_channels_detected' in locals() else 3
        if loaded_state and "vq.embedding.weight" in loaded_state:
            from modules.spatial_model import DiscreteVQVAE
            num_embeddings = loaded_state["vq.embedding.weight"].shape[0]
            encoder = DiscreteVQVAE(latent_dim=latent_dim_detected, model_size=model_size_detected, input_spatial_dim=input_spatial_dim_detected, in_channels=in_channels_detected, num_embeddings=num_embeddings).to(device)
        else:
            encoder = TinyVAE(latent_dim=latent_dim_detected, model_size=model_size_detected, input_spatial_dim=input_spatial_dim_detected, in_channels=in_channels_detected).to(device)
        
        if loaded_state:
            try:
                encoder.load_state_dict(loaded_state)
                logger.debug(f"Loaded Pretrained VAE from {vae_path}")
            except Exception as e:
                logger.error(f"Failed to load state dict (Size mismatch?): {e}")
                raise e

        # DrQ: Encoder is trained end-to-end
        encoder.train()
        
        # Removed Precompute All Latents ONCE: DrQ trains vision end-to-end dynamically.

    # Prepare Dataset (Now Instant thanks to pre-computed latents)
    if progress_callback: progress_callback(0, 0.1) # Signal data prep start
    logger.debug("Constructing Dataset Pairs...")
    
    codebook_tensor = None
    if loaded_state and "vq.embedding.weight" in loaded_state:
        codebook_tensor = loaded_state["vq.embedding.weight"].to(device)
        
    dataset = SpatialCQLDataset(trajectories, data_root, device, progress_callback, dataset_percent=dataset_percent, goal_type=goal_type, image_size=input_spatial_dim_detected, codebook=codebook_tensor)
    if progress_callback: progress_callback(0, 1.0) # Signal data prep done
    
    if not train_from_scratch and encoder is not None and not is_groundtruth:
        dataset.precompute_latents(encoder, device)
    elif is_groundtruth:
        dataset.use_precomputed = True
        
    inject_telemetry = use_telemetry and os.path.exists(os.path.join(data_root, "master_telemetry.csv")) and not is_groundtruth
    if inject_telemetry:
        import pandas as pd
        import math
        logger.info("Injecting Telemetry vectors alongside visual latents.")
        telemetry_dict = {}
        telemetry_path = os.path.join(data_root, "master_telemetry.csv")
        df = pd.read_csv(telemetry_path)
        for _, row in df.iterrows():
            yaw_rad = math.radians(row['yaw_deg'])
            telemetry_dict[str(row['ts'])] = torch.tensor([
                row['cx'] / 640.0,
                row['cy'] / 480.0,
                math.cos(yaw_rad),
                math.sin(yaw_rad)
            ], dtype=torch.float)
            
        unique_nodes_count = 0
        visited = set()
        for traj_idx, traj in enumerate(trajectories):
            for node in traj:
                if id(node) in visited: continue
                visited.add(id(node))
                if 'latent' in node: # Should be there after precompute_latents
                    ts = str(node.get('timestamp', ''))
                    
                    if ts == '':
                        path = node.get('image_path', '')
                        if path:
                            base = os.path.basename(path)
                            ts = base.replace('frame_', '').replace('webcam_frame_', '').replace('.jpg', '')
                            
                    if ts in telemetry_dict:
                         node['latent'] = torch.cat([node['latent'].to('cpu'), telemetry_dict[ts].to('cpu')])
                         unique_nodes_count += 1
                    else:
                         node['latent'] = torch.cat([node['latent'].to('cpu'), torch.zeros(4, dtype=torch.float)])
        
        logger.debug(f"Successfully injected telemetry into {unique_nodes_count} physics latents.")
        latent_dim_detected += 4
    
    # [FIX] HER Batch Forcing for Sparse Rewards
    # Instead of balancing Actions broadly, we force the batch to contain 25% positive (+50) reward transitions
    # out of the ~550 we identified in the 115k dataset.
    if goal_type == 'discrete_exact' and getattr(dataset, 'exact_latent', None) is not None:
        logger.info("Executing HER Batch Forcing: Identifying Terminal States for exact matches...")
        all_terminal_indices = []
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            latent_next_stack = sample.get('next_nodes')
            if latent_next_stack:
                latent_tensor = latent_next_stack[-1].get('latent')
                if latent_tensor is not None:
                    vision_dim = dataset.exact_latent.shape[-1]
                    vision_latent = latent_tensor[:vision_dim].view(-1).to(device)
                    dists = torch.norm(dataset.exact_latents.to(device) - vision_latent.unsqueeze(0), dim=1)
                    if torch.any(dists < 0.001):
                        all_terminal_indices.append(idx)
        
        if all_terminal_indices:
            terminal_indices = all_terminal_indices
            # Tag the dataset so __getitem__ knows exactly which indices are terminal without re-evaluating
            dataset.terminal_indices_set = set(terminal_indices)
            
            num_terminal = len(terminal_indices)
            num_total = len(dataset)
            num_negative = num_total - num_terminal
            
            w_positive = (num_negative / num_terminal) * (0.25 / 0.75)
            
            sample_weights = [1.0] * num_total
            for idx in terminal_indices:
                sample_weights[idx] = w_positive
                
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_total, replacement=True)
            logger.info(f"Enabled Batch Forcing! {num_terminal} exact Terminal frames assigned {w_positive:.2f}x sampling weight.")
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        else:
            logger.warning("No Terminal Rewards found. Reverting to uniform shuffling.")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
    elif goal_type == 'group_goal' and getattr(dataset, 'group_centroid', None) is not None:
        logger.info("Executing HER Batch Forcing: Identifying Terminal States...")
        all_dists = []
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            latent_next_stack = sample.get('next_nodes')
            if latent_next_stack:
                latent_tensor = latent_next_stack[-1].get('latent')
                if latent_tensor is not None:
                    vision_dim = dataset.group_centroid.shape[-1]
                    vision_latent = latent_tensor[:vision_dim]
                    dist = torch.norm(vision_latent.view(-1).to(device) - dataset.group_centroid).item()
                    all_dists.append((idx, dist))
                    
        # [FIX] Strictly enforce the user's explicit group_avg_dist threshold
        # rather than dynamically overriding it to hit a 1000-state quota
        threshold = dataset.group_avg_dist * 1.1
        terminal_indices = [idx for idx, dist in all_dists if dist < threshold]
        logger.info(f"Found {len(terminal_indices)} terminal states using explicit threshold {threshold:.4f}")
        
        if terminal_indices:
            num_terminal = len(terminal_indices)
            num_total = len(dataset)
            num_negative = num_total - num_terminal
            
            # We want terminal states to represent exactly 25% of the batch
            w_positive = (num_negative / num_terminal) * (0.25 / 0.75)
            
            sample_weights = [1.0] * num_total
            for idx in terminal_indices:
                sample_weights[idx] = w_positive
                
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_total, replacement=True)
            logger.info(f"Enabled Batch Forcing! {num_terminal} Terminal frames assigned {w_positive:.2f}x sampling weight.")
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        else:
            logger.warning("No Terminal Rewards found. Reverting to uniform shuffling.")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        logger.debug("Using standard shuffling (Uniform distribution) for accurate Q-value backpropagation.")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # The previous VAE initialization logic was hoisted to pre-compute latents above.
        
    # Policy Input = Latent Dim [Current] * 1 (Fixed-Goal Only)
    # [REMOVED] Optical Flow Buffer (frame stacking)
    # [REMOVED] Backward Compatibility & Goal Conditioning
    policy_input_dim = latent_dim_detected
    if getattr(dataset, 'goal_type', None) == 'discrete_exact' and codebook_tensor is not None:
        policy_input_dim = codebook_tensor.shape[0]
        
    logger.debug(f"Initializing CQL Policy for VAE with Latent={latent_dim_detected} (Input={policy_input_dim})")
    
    # [MODIFIED] Disable LayerNorm for Discrete VQ-VAE to preserve 100x magnification
    use_ln = True
    if getattr(dataset, 'goal_type', None) == 'discrete_exact':
        use_ln = False
    
    policy = CQLNetwork(input_dim=policy_input_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, model_size=model_size, use_ln=use_ln).to(device)
    
    q_net = FullQNet(encoder, policy, has_goal=False).to(device)
    target_q_net = copy.deepcopy(q_net) # Copy full structure
    
    # Optimizer: Only train adapter and policy
    # If not end-to-end, freeze encoder cleanly natively
    if not train_from_scratch and encoder is not None:
        for param in encoder.parameters():
            param.requires_grad = False
            
    params_to_train = [
        {'params': policy.parameters()},
        {'params': encoder.parameters() if hasattr(encoder, 'parameters') else []}
    ]
    optimizer = optim.Adam(params_to_train, lr=learning_rate)
    
    q_net.train() # This sets training mode. 
    # But backbone is frozen via requires_grad=False if not train_from_scratch.
    # Usually we want eval() for frozen backbone to keep statistics fixed if pre-trained.
    if q_net.encoder is not None and not train_from_scratch:
        q_net.encoder.eval() # Ensure VAE is in eval mode (BatchNorm/Dropout if any, though TinyVAE has none)
    
    logger.info(f"Starting Training Loop for {num_epochs} epochs...")
    if device.type == 'cuda':
        init_vram = torch.cuda.memory_allocated(device) / (1024 * 1024)
        logger.info(f"Initial VRAM usage: {init_vram:.0f} MB")
        
    # [FIX] Force UI to unlock its "ignore stale state" check by emitting epoch <= 1.0
    if progress_callback:
        progress_callback(0, 0)
    
    last_report_time = 0
    
    for epoch in range(num_epochs):
        # Stop Check
        if stop_event and stop_event.is_set():
            logger.info("Training interrupted by user.")
            break

        total_loss = 0
        num_batches = len(dataloader)
        
        for i, (latent_curr_stack, latent_next_stack, latent_goal, state_curr, state_next, action, reward, done) in enumerate(dataloader):
            # Intra-batch stop check
            if stop_event and stop_event.is_set():
                break
                
            latent_curr_stack = latent_curr_stack.to(device)
            latent_next_stack = latent_next_stack.to(device)
            latent_goal = latent_goal.to(device)
            state_curr = state_curr.to(device)
            state_next = state_next.to(device)
            action = action.to(device)
            reward = reward.to(device)
            done = done.to(device)
            
            # Rely strictly on the dataset HER rewards!
            reward = reward.unsqueeze(1) if reward.dim() == 1 else reward

            if train_from_scratch:
                # --- DrQ Image Augmentations ---
                try:
                    from utils.augmentations import random_shift
                except:
                    # inline fallback
                    def random_shift(x, pad=4): return x

                B, S, C, H, W = latent_curr_stack.shape
                img_curr_flat = latent_curr_stack.view(B*S, C, H, W)
                img_next_flat = latent_next_stack.view(B*S, C, H, W)
                
                img_curr_aug = random_shift(img_curr_flat).view(B, S, C, H, W)
                img_next_aug = random_shift(img_next_flat).view(B, S, C, H, W)
                img_goal_aug = random_shift(latent_goal)

                # --- End-To-End Forward Pass ---
                state_input_curr = q_net(img_curr_aug, img_goal_aug, state_curr) # Actually q_values, but variable name reused for compatibility
                state_input_next = target_q_net(img_next_aug, img_goal_aug, state_next)
                # -----------------------------------------------
            else:
                # --- Standard CQL Forward Pass ---
                B, S, D = latent_curr_stack.shape
                mu_curr = latent_curr_stack.view(B, -1)   # [B, S*D]
                mu_next = latent_next_stack.view(B, -1)   # [B, S*D]
                
                if getattr(q_net, 'has_goal', True):
                    state_combined_curr = torch.cat([mu_curr, latent_goal], dim=1)
                    state_combined_next = torch.cat([mu_next, latent_goal], dim=1)
                else:
                    state_combined_curr = mu_curr
                    state_combined_next = mu_next

                # Bypass q_net's vision backbone
                state_input_curr = q_net.policy(state_combined_curr)
                state_input_next = target_q_net.policy(state_combined_next)
            
            # Current Q: (Batch, ActionDim)
            # print(f"DEBUG: Processing batch {i} with latent_curr shape={latent_curr_stack.shape}, state_curr shape={state_curr.shape}")
            q_values = state_input_curr
            q_action = q_values.gather(1, action.view(-1, 1))

            # Target Q
            with torch.no_grad():
                # Double DQN / Standard DQN
                next_q_values = state_input_next
                max_next_q = next_q_values.max(1, keepdim=True)[0]
                target_q = reward.clone() + GAMMA * (1 - done.view(-1, 1)) * max_next_q
                
            # TD Loss
            td_loss = F.mse_loss(q_action, target_q)
            
            # CQL Loss
            logsumexp_q = torch.logsumexp(q_values, dim=1, keepdim=True)
            cql_loss = (logsumexp_q - q_action).mean()
            
            loss = td_loss + alpha * cql_loss
            
            # [FIX] Discrete Orthogonal Bias Bleed
            # Because Action 5 is oversampled to 25%, its global bias inflates. Unpropagated sparse nodes
            # evaluate navigation actions as 0, so Action 5 wins purely through the global bias.
            if getattr(dataset, 'goal_type', None) == 'discrete_exact':
                penalty_5 = ((1.0 - done.view(-1)) * F.relu(q_values[:, 5] + 0.1)).mean()
                loss = loss + 1.0 * penalty_5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # [NEW] Dynamic interval (Time-based ~1Hz)
            current_time = time.time()
            if (current_time - last_report_time) >= 1.0 or i == 0 or (i + 1) == num_batches:
                last_report_time = current_time
                
                # logger.info(f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{num_batches} | Loss: {loss.item():.4f}")
                if progress_callback:
                    # Plot fractional epoch for finer granularity
                    current_step = epoch + (i + 1) / num_batches
                    progress_callback(current_step, loss.item())
            
        avg_loss = total_loss / max(1, len(dataloader))
        vram_msg = ""
        if device.type == 'cuda':
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            vram_msg = f" | Peak VRAM: {vram_mb:.0f} MB"
        logger.info(f"Epoch {epoch+1}/{num_epochs} COMPLETE, Avg Loss: {avg_loss:.4f}{vram_msg}")
        
        # Dial home
        if progress_callback:
            progress_callback(epoch + 1, avg_loss)
        
        # Target Update
        target_q_net.load_state_dict(q_net.state_dict())
        
    if model_filename:
        # Ensure it ends with .pth
        if not model_filename.endswith(".pth"):
            model_filename += ".pth"
        policy_basename = model_filename.replace(".pth", "")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if vae_path:
            vae_basename = os.path.splitext(os.path.basename(vae_path))[0]
            
            if goal_type == 'dark_wall_seek':
                policy_basename = f"{vae_basename}-dark-wall-cql_{timestamp}"
            elif goal_type == 'group_goal':
                policy_basename = f"{vae_basename}-group-goal-cql_{timestamp}"
            else:
                policy_basename = f"{vae_basename}-cql_{timestamp}"
        else:
            parent_name = os.path.basename(os.path.normpath(data_root))
            
            modifier = ""
            if goal_type == 'dark_wall_seek':
                modifier = "-dark-wall"
                
            if not parent_name or parent_name == 'data' or parent_name == '.':
                 policy_basename = f"cql_policy{modifier}_{timestamp}"
            else:
                 policy_basename = f"{parent_name}{modifier}-cql_{timestamp}"
    
    policy_path = os.path.join(data_root, f"{policy_basename}.pth")

    # torch.save(q_net.encoder.state_dict(), encoder_path) # Skip saving encoder separately if we are trusting the VAE file source of truth
    
    # [NEW] Save Policy with Metadata (Valid Actions)
    valid_actions_list = sorted(list(dataset.valid_actions)) if dataset.valid_actions else [1, 2, 3, 4]
    policy_save_dict = {
        'model_state_dict': q_net.policy.state_dict(),
        'valid_actions': valid_actions_list,
        'hyperparameters': {
            'cql_alpha': alpha,
            'model_size': model_size,
            'threshold_multiplier': 1.1,
            'sampling_multiplier': (0.25 / 0.75)
        }
    }
    
    # [NEW] Group Goal topological embedding
    if getattr(dataset, 'goal_type', None) == 'group_goal':
        if getattr(dataset, 'group_centroid', None) is not None:
            policy_save_dict['group_centroid'] = dataset.group_centroid.cpu().numpy().tolist()
        if getattr(dataset, 'group_avg_dist', None) is not None:
            policy_save_dict['group_avg_dist'] = dataset.group_avg_dist
            
    if train_from_scratch and q_net.encoder is not None:
        policy_save_dict['encoder_state_dict'] = q_net.encoder.state_dict()
        
    torch.save(policy_save_dict, policy_path)
    
    logger.info(f"Models saved: {policy_path} (Valid Actions: {valid_actions_list}, E2E Vision State Included: {train_from_scratch})")
    
    # Goals generation was moved to VAE training specifically.

    # --- POST-TRAINING VERIFICATION ---
    eval_path = evaluate_policy(q_net, dataset, trajectories, device)
    
    # If we didn't save a new encoder, we return the VAE path we loaded as the "encoder path"
    final_encoder_path = vae_path
    
    return final_encoder_path, policy_path, eval_path

def evaluate_policy(q_net, dataset, trajectories, device, num_samples=500):
    """
    Runs a smoke test on the trained model using random samples from the dataset.
    Detects Policy Collapse (e.g., outputting only STOP).
    """
    logger.info("-" * 40)
    logger.info("POST-TRAINING VERIFICATION")
    logger.info("-" * 40)
    
    q_net.eval()
    actions_count = {}
    
    # Random Sampling
    indices = np.random.choice(len(dataset), size=min(len(dataset), num_samples), replace=False)
    
    logger.info(f"Running inference on {len(indices)} random samples...")
    
    with torch.no_grad():
        for idx in indices:
            latent_curr_stack, _, latent_goal, state_curr, _, _, _, _ = dataset[idx]
            
            # Prepare Batch (1)
            latent_curr_stack = latent_curr_stack.unsqueeze(0).to(device)
            latent_goal = latent_goal.unsqueeze(0).to(device)
            state_curr = state_curr.unsqueeze(0).to(device)
            
            # Assemble state directly (64-D Latent Topology or Images)
            if latent_curr_stack.dim() == 3: # Precomputed vectors: [Batch, Stack, LatentDim]
                B, S, D = latent_curr_stack.shape
                mu_curr = latent_curr_stack.view(B, -1)
                if getattr(q_net, 'has_goal', True):
                    state_combined = torch.cat([mu_curr, latent_goal], dim=1)
                else:
                    state_combined = mu_curr
                q_values = q_net.policy(state_combined)
            else:
                q_values = q_net(latent_curr_stack, latent_goal, state_curr)
            
            # Mask (if needed, but here we test raw model preference)
            # If we wanted to match planner logic perfectly we'd apply mask, 
            # but raw preference tells us training health better.
            
            action = torch.argmax(q_values, dim=1).item()
            actions_count[action] = actions_count.get(action, 0) + 1

    # Report
    logger.info("Predicted Action Distribution:")
    action_map = {0: "STOP", 1: "FWD", 2: "REV", 3: "HRD_L", 4: "HRD_R", 5: "SFT_L", 6: "SFT_R"}
    
    max_pct = 0
    dom_action = -1
    
    for k in sorted(actions_count.keys()):
        count = actions_count[k]
        pct = (count / len(indices)) * 100
        name = action_map.get(k, str(k))
        logger.info(f"  {name:<6} (ID {k}): {count} ({pct:.1f}%)")
        
        if pct > max_pct:
            max_pct = pct
            dom_action = k
            
    # Judgment
    if max_pct >= 95.0:
        logger.error(f"FAIL: Policy Collapse Detected! Model chose {action_map.get(dom_action, dom_action)} {max_pct}% of the time.")
    elif max_pct >= 80.0:
        logger.warning(f"WARNING: Low Diversity. Dominant action: {action_map.get(dom_action, dom_action)} ({max_pct}%)")
    else:
        logger.info("PASS: Policy seems to have learned a distribution.")
        
    logger.info("Evaluating sequence adherence using Confusion Matrix...")
    
    y_true = []
    y_pred = []
    
    # Fall back to evaluating a random subset of all valid transitions
    # This ensures we see the true distribution of all 9 classes, including STOP and VEERs
    eval_indices = np.random.choice(len(dataset), size=min(len(dataset), num_samples), replace=False)
    with torch.no_grad():
        for idx in eval_indices:
            # __getitem__ returns:
            # latent_curr_stack, latent_next_stack, latent_goal, state_curr, state_next, action, reward, done
            latent_curr_stack, _, latent_goal, state_curr, _, true_action, _, _ = dataset[idx]
            
            latent_curr_stack = latent_curr_stack.unsqueeze(0).to(device)
            latent_goal = latent_goal.unsqueeze(0).to(device)
            state_curr = state_curr.unsqueeze(0).to(device)

            if latent_curr_stack.dim() == 3: # Precomputed vectors: [Batch, Stack, LatentDim]
                B, S, D = latent_curr_stack.shape
                mu_curr = latent_curr_stack.view(B, -1)
                if getattr(q_net, 'has_goal', True):
                    state_combined = torch.cat([mu_curr, latent_goal], dim=1)
                else:
                    state_combined = mu_curr
                q_values = q_net.policy(state_combined)
            else:
                q_values = q_net(latent_curr_stack, latent_goal, state_curr)
            pred_action = torch.argmax(q_values, dim=1).item()
            y_true.append(true_action.item())
            y_pred.append(pred_action)
            
    # Generate Confusion Matrix
    # We only care about the actions the model was actually trained on (WASD)
    valid_actions_list = sorted(list(dataset.valid_actions)) if dataset.valid_actions else [1, 3, 4]
    
    # Filter the action map specifically to the active IDs
    cm_labels_ids = valid_actions_list
    filtered_action_labels = [action_map.get(k, str(k)) for k in cm_labels_ids]
    
    logger.info("\n" + "="*50 + "\n📊 CLASSIFICATION REPORT\n" + "="*50)
    report = classification_report(y_true, y_pred, labels=cm_labels_ids, target_names=filtered_action_labels, zero_division=0)
    for line in report.split('\n'):
        logger.info(line)
        
    conf_matrix = confusion_matrix(y_true, y_pred, labels=cm_labels_ids)
    
    cm_path = None
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=filtered_action_labels, yticklabels=filtered_action_labels, ax=ax)
        
        ax.set_title('Evaluation Confusion Matrix', fontsize=14)
        ax.set_ylabel('Actual Action', fontsize=12)
        ax.set_xlabel('Predicted Action', fontsize=12)
        
        # Save the plot
        # Try to find the associated policy path to save the image next to it
        data_root = dataset.data_root
        import glob
        # Find newest policy to pair with
        candidates = glob.glob(os.path.join(data_root, "*cql_*.pth")) + glob.glob(os.path.join(data_root, "cql_policy*.pth"))
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            newest_policy = candidates[0]
            basename = os.path.splitext(os.path.basename(newest_policy))[0]
            cm_path = os.path.join(data_root, f"{basename}_confusion.png")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved Confusion Matrix to {cm_path}")
            
        plt.close(fig)
    except Exception as e:
        import traceback
        logger.error(f"Failed to generate or save confusion matrix plot: {e}")
        traceback.print_exc()
        
    # === TERMINAL GOAL-CLEARING EVALUATION ===
    if getattr(dataset, 'goal_type', None) == 'group_goal':
        logger.info("\n" + "="*50 + "\n🎯 TERMINAL GOAL CLEARING EVALUATION\n" + "="*50)
        
        terminal_indices = []
        for idx in range(len(dataset)):
            _, _, _, _, _, _, reward, _ = dataset[idx]
            if reward.item() >= 49.0: # Match +50
                terminal_indices.append(idx)
        
        if len(terminal_indices) > 0:
            logger.info(f"Identified {len(terminal_indices)} goal-clearing terminal transitions.")
            
            t_correct = 0
            t_total = len(terminal_indices)
            t_actions_count = {}
            t_true_actions_count = {}
            
            with torch.no_grad():
                for idx in terminal_indices:
                    latent_curr_stack, _, latent_goal, state_curr, _, true_action, _, _ = dataset[idx]
                    
                    latent_curr_stack = latent_curr_stack.unsqueeze(0).to(device)
                    latent_goal = latent_goal.unsqueeze(0).to(device)
                    state_curr = state_curr.unsqueeze(0).to(device)

                    if latent_curr_stack.dim() == 3: 
                        B, S, D = latent_curr_stack.shape
                        mu_curr = latent_curr_stack.view(B, -1)
                        if getattr(q_net, 'has_goal', True):
                            state_combined = torch.cat([mu_curr, latent_goal], dim=1)
                        else:
                            state_combined = mu_curr
                        q_values = q_net.policy(state_combined)
                    else:
                        q_values = q_net(latent_curr_stack, latent_goal, state_curr)
                        
                    pred_action = torch.argmax(q_values, dim=1).item()
                    t_true_action = true_action.item()
                    
                    if pred_action == t_true_action:
                        t_correct += 1
                        
                    t_actions_count[pred_action] = t_actions_count.get(pred_action, 0) + 1
                    t_true_actions_count[t_true_action] = t_true_actions_count.get(t_true_action, 0) + 1
                    
            t_acc = (t_correct / t_total) * 100
            
            logger.info(f"Terminal Action Accuracy: {t_acc:.1f}% ({t_correct}/{t_total} correct)")
            logger.info("Actions PREDICTED for terminal states:")
            for k, v in t_actions_count.items():
                logger.info(f"  {action_map.get(k, str(k))}: {v} ({v/t_total*100:.1f}%)")
            logger.info("Actual TRUE actions targeting goal:")
            for k, v in t_true_actions_count.items():
                logger.info(f"  {action_map.get(k, str(k))}: {v} ({v/t_total*100:.1f}%)")
        else:
            logger.warning("No terminal goal-clearing boundaries found in dataset.")

    logger.info("-" * 40)
    
    return cm_path


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_DIR

    parser = argparse.ArgumentParser(description="Train Goal-Conditioned CQL Navigation Policy")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.2, help="CQL alpha (conservatism weight)")
    parser.add_argument("--model_size", type=str, default="large", help="Network size (small/large)")
    parser.add_argument("--dataset_percent", type=int, default=100, help="Percentage of dataset to use")
    parser.add_argument("--goal_type", type=str, default="her", help="Goal type (her, group_goal, discrete_exact, dark_wall_seek)")
    parser.add_argument("--vae_model", type=str, default=None, help="VAE model filename (auto-detected if omitted)")
    args = parser.parse_args()

    train(
        data_root=DATA_DIR,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        alpha=args.alpha,
        model_size=args.model_size,
        dataset_percent=args.dataset_percent,
        goal_type=args.goal_type,
        vae_model_filename=args.vae_model,
    )
