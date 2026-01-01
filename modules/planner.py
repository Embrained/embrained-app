
import os
import glob
import sys
import importlib.util
import logging
import torch
import numpy as np
import random
import time
from config import MODELS_DIR, GOAL_SWITCH_INTERVAL
from modules.spatial_model import CQLNetwork

class Planner:
    def __init__(self, device='cpu', model_path=None):
        self.device = torch.device(device)
        self.model_name = "CQLNetwork"
        self.policy = self._load_policy(model_path)
        self.policy.to(self.device).eval() 
        
        self.goals = self._load_goals() # List of Keypoints arrays (64,)
        self.current_goal_idx = 0
        self.last_goal_switch = 0
        self.stop_cooldown_end_time = 0
        
        logging.info(f"Planner initialized. Loaded {len(self.goals)} goals.")

    def load_model(self, model_path):
        """
        Loads a specific model file into the policy.
        """
        logging.info(f"Loading policy weights from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Check if it's a full checkpoint or just state_dict
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.policy.load_state_dict(state_dict)
            self.policy.eval()
            self.model_name = os.path.basename(model_path)
            logging.info(f"Successfully loaded {self.model_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model {model_path}: {e}")
            return False

    def _load_policy(self, specific_path=None):
        """
        Loads the Discrete CQL Policy (MLP).
        """
        logging.info("Loading CQLNetwork Policy...")
        # 128 input (64+64), 256 hidden, 5 output
        # ACTION_DIM in train_cql.py = 5
        policy = CQLNetwork(input_dim=128, hidden_dim=256, action_dim=5)
        
        candidates = []
        if specific_path:
            candidates.append(specific_path)
            
        # Look for weights
        candidates.extend([
             os.path.join(MODELS_DIR, "cql_policy.pth"),
             os.path.join(os.path.dirname(MODELS_DIR), "data", "cql_policy.pth"),
             os.path.join(MODELS_DIR, "..", "data", "cql_policy.pth")
        ])
        
        weights_loaded = False
        for path in candidates:
             if os.path.exists(path):
                 logging.info(f"Found policy weights at {path}")
                 try:
                     state_dict = torch.load(path, map_location=self.device)
                     policy.load_state_dict(state_dict)
                     weights_loaded = True
                     break
                 except Exception as e:
                     logging.warning(f"Failed to load candidate {path}: {e}")
        
        if not weights_loaded:
             logging.warning("No cql_policy.pth found! Using random weights.")
             
        return policy

    def _load_goals(self):
        """
        Load goal keypoints.
        """
        # Look in data dir mostly as train_cql saves there
        candidates = [
             os.path.join(MODELS_DIR, 'goals.npy'),
             os.path.join(MODELS_DIR, '..', 'data', 'goals.npy')
        ]
        
        # Add recursive search for best goals
        try:
             import glob
             from config import DATA_DIR
             search_pattern = os.path.join(DATA_DIR, "**", "goals.npy")
             found = glob.glob(search_pattern, recursive=True)
             if found:
                 found.sort(key=os.path.getmtime, reverse=True)
                 candidates.insert(0, found[0])
        except: pass
        
        for path in candidates:
            if os.path.exists(path):
                return np.load(path)
                
        logging.warning("No goals.npy found. Creating dummy goals.")
        return [np.random.randn(64).astype(np.float32) for _ in range(5)]

    def decide(self, z_current, dist_threshold=None):
        """
        Input: z_current (64,) Keypoints
        Output: (action_id, distance_to_goal, goal_index)
        """
        # 1. Update Goal State (Rotation)
        now = time.time()
        if now - self.last_goal_switch > GOAL_SWITCH_INTERVAL:
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.goals)
            self.last_goal_switch = now
            logging.info(f"Switched to Goal Index {self.current_goal_idx}")
            
        z_goal = self.goals[self.current_goal_idx]
        
        # 2. Distance Check (Euclidean distance in Keypoint space?)
        # Keypoints are normalized coordinates (-1 to 1). 
        # Euclidean distance is a reasonable metric for "closeness" in feature space.
        distance = np.linalg.norm(z_current - z_goal)
        
        # 3. Policy Inference
        # Concatenate: [Current (64), Goal (64)] -> (128,)
        obs = np.concatenate([z_current, z_goal])
        
        # Tensorize
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device) # (1, 128)
        
        with torch.no_grad():
            q_values = self.policy(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        # 4. Stop Logic (Reflex)
        from config import STOP_DISTANCE_THRESHOLD, STOP_COOLDOWN_S
        
        threshold = dist_threshold if dist_threshold is not None else STOP_DISTANCE_THRESHOLD
        
        if distance < threshold:
            # action = 0 # STOP is 0 in train_cql.py
            action = 0 
            self.stop_cooldown_end_time = now + STOP_COOLDOWN_S
        elif hasattr(self, 'stop_cooldown_end_time') and now < self.stop_cooldown_end_time:
            action = 0
            
        return action, distance, self.current_goal_idx
