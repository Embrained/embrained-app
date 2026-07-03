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
import sys
import importlib.util
import logging
import logging
from backend.utils import safe_import_torch
torch = safe_import_torch()

import numpy as np
import random
import time
from config import MODELS_DIR, HIDDEN_DIM, INPUT_DIM
from modules.spatial_model import CQLNetwork
import random

class RandomCQLPolicy:
    """A dummy policy that acts as a random Markov walk substitute during identical CQL evaluation regimes."""
    def __init__(self):
        pass
    def eval(self):
        pass
    def to(self, device):
        return self
    def __call__(self, x):
        import torch
        import random
        # FixedGoal BC network outputs 4 actions (0: FWD, 1: REV, 2: LEFT, 3: RIGHT)
        # We want to randomly pick between FWD (0), LEFT (2), RIGHT (3) to avoid reversing constantly
        q_vals = torch.zeros((1, 4), dtype=torch.float32)
        chosen_action = random.choice([0, 2, 3])
        q_vals[0, chosen_action] = 1.0 # Set the chosen action's Q-value to be the max
        return q_vals

class Planner:
    """
    Planner module for policy-based navigation.
    
    When loaded with a CQL policy, it attempts to navigate towards the current goal latent (32-dim).
    """
    def __init__(self, device='cpu', model_path=None):
        self.device = torch.device(device)
        self.model_name = "None"
        self.valid_actions = None # [NEW] Mask for inference
        self.policy = None

        if model_path:
            self.policy = self._load_policy(model_path)
            self.policy.to(self.device).eval()
        else:
             pass
        
        self.goals = [] # List of dicts: {'latent': np.array, 'image_path': str}
        self.mu_goal = None
        self.goal_radius = 0.0
        self.z_smoothed = None
        self.last_q_value = None
        
        self.current_goal_idx = 0
        self.last_goal_switch = 0
        self.stop_cooldown_end_time = 0
        self.consecutive_reverses = 0
        
        # We don't load goals here anymore; we rely on the engine to call load_goals_for_vae

    def _compute_goal_envelope(self):
        """Calculates the centroid and bounds of the goal region."""
        m_name = getattr(self, 'model_name', '')
        if m_name and ('group-goal' in m_name or 'group_goal' in m_name or 'fixed_goal' in m_name or 'discrete_cql' in m_name):
             return # Never overwrite natively embedded model checkpoint topological configurations
             
        if not self.goals:
            self.mu_goal = None
            self.goal_radius = 0.0
            return
            
        latents = [g['latent'] for g in self.goals]
        latents_arr = np.array(latents) # (N, 32)
        
        self.mu_goal = np.mean(latents_arr, axis=0)
        
        # Calculate radius (max L2 norm from centroid)
        if len(latents) > 1:
            distances = np.linalg.norm(latents_arr - self.mu_goal, axis=1)
            self.goal_radius = np.max(distances)
            self.goal_avg_dist = np.mean(distances)
        else:
            self.goal_radius = 0.0
            self.goal_avg_dist = 0.0


    def reload_goals(self):
        """Re-reads goal latents from models/goals.npy or fallback."""
        logging.info("Planner: Reloading goals...")
        self.goals = self._load_goals()
        self._compute_goal_envelope()
        self.current_goal_idx = 0
        self.last_goal_switch = time.time()

    def set_dynamic_goal(self, goal_latent, image_path=None):
        """Overrides the current rotation with a dynamically injected latent."""
        self.goals = [{'latent': goal_latent, 'image_path': image_path}]  # Only one goal in rotation now
        self._compute_goal_envelope()
        self.current_goal_idx = 0
        self.last_goal_switch = time.time()
        logging.info("Planner: Dynamically injected new goal latent.")

    def load_model(self, model_path):
        """
        Loads a specific model file into the policy.
        Handles architecture mismatches (LayerNorm vs Legacy).
        """
        logging.info(f"Loading policy weights from {model_path}")
        if not torch:
             return False
             
        if model_path is None:
            self.policy = None
            self.model_name = "None"
            return True
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint
            
            # Check if it's a full checkpoint dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'valid_actions' in checkpoint:
                    self.valid_actions = checkpoint['valid_actions']
                    logging.info(f"Loaded Valid Actions Mask: {self.valid_actions}")
                    
                if 'hyperparameters' in checkpoint:
                    self.hyperparameters = checkpoint['hyperparameters']
                    
                # [NEW] Check for native geographic payload (Group Goal embedded centroid)
                if 'group_centroid' in checkpoint:
                    self.mu_goal = np.array(checkpoint['group_centroid'], dtype=np.float32)
                    self.goal_avg_dist = checkpoint.get('group_avg_dist', 0.0)
                    logging.info("Extracted embedded Group Goal geographic configuration natively from checkpoint.")
                    
                # [NEW] Extract sculpted DrQ filters for the vision system
                if 'encoder_state_dict' in checkpoint and hasattr(self, 'vision') and self.vision:
                    try:
                        self.vision.model.load_state_dict(checkpoint['encoder_state_dict'])
                        logging.info("Successfully bound end-to-end DrQ encoder filters.")
                    except Exception as e:
                        logging.error(f"Failed to bind DrQ encoder: {e}")
            
            # Create a FRESH policy instance (Default: use_ln=True)
            # This ensures we don't carry over legacy state if switching from old -> new
            
            # [MODIFIED] Dynamic Size Loading Loop for Explicit Load
            possible_sizes = ['tiny', 'small', 'medium', 'large', 'enormous', 'tectonic']
            
            is_discrete = 'discrete' in os.path.basename(model_path).lower() and 'cve' not in os.path.basename(model_path).lower()
            possible_input_dims = [512] if is_discrete else [32] 
            
            loaded_policy = None
            loaded_ok = False
            
            # [NEW] Check for Markov Random Walk Baseline
            if "oracle_control.pth" in os.path.basename(model_path):
                loaded_policy = RandomCQLPolicy()
                self.model_name = os.path.basename(model_path)
                logging.info(f"Intercepted Markov Random Walk Baseline evaluation! Emitting dummy policy.")
                loaded_ok = True
            
            # [NEW] Check for Telemetry Oracle
            elif "telemetry_oracle" in os.path.basename(model_path):
                try:
                    from config import ACTION_DIM
                    from modules.spatial_model import OracleQNetwork
                    test_policy = OracleQNetwork(state_dim=4, goal_dim=4, action_dim=ACTION_DIM)
                    test_policy.to(self.device)
                    test_policy.load_state_dict(state_dict, strict=False)
                    loaded_policy = test_policy
                    self.model_name = os.path.basename(model_path)
                    logging.info(f"Successfully loaded OracleQNetwork (Input=8, Actions={ACTION_DIM}) from {model_path}")
                    loaded_ok = True
                except Exception as e:
                    logging.error(f"Failed to load OracleQNetwork: {e}")
            elif "ir_reflex" in os.path.basename(model_path):
                try:
                    from modules.spatial_model import IRPredictorNetwork
                    
                    # Dynamically read tensor size
                    try:
                        input_shape = state_dict['input_layer.weight'].shape[1] # e.g. 392
                        extracted_state_dim = input_shape - 8 # e.g. 384
                    except Exception:
                        extracted_state_dim = 96
                        
                    test_policy = IRPredictorNetwork(state_dim=extracted_state_dim, action_dim=3)
                    test_policy.to(self.device)
                    test_policy.load_state_dict(state_dict, strict=False)
                    loaded_policy = test_policy
                    self.model_name = os.path.basename(model_path)
                    logging.info(f"Successfully loaded IRPredictorNetwork (Dim={extracted_state_dim}) from {model_path}")
                    loaded_ok = True
                except Exception as e:
                    logging.error(f"Failed to load IRPredictorNetwork: {e}")
            elif "fixed_goal" in os.path.basename(model_path) and "cql" not in os.path.basename(model_path).lower():
                try:
                    from modules.spatial_model import FixedGoalBCNetwork
                    from config import ACTION_DIM
                    
                    # Dynamically read tensor size
                    try:
                        input_shape = state_dict['input_layer.weight'].shape[1] # e.g. 384
                        extracted_state_dim = input_shape // 3 # internal Network multiplies by 3
                    except Exception:
                        extracted_state_dim = 32
                        
                    test_policy = FixedGoalBCNetwork(state_dim=extracted_state_dim, action_dim=4)
                    test_policy.to(self.device)
                    test_policy.load_state_dict(state_dict, strict=False)
                    loaded_policy = test_policy
                    self.model_name = os.path.basename(model_path)
                    
                    logging.info(f"Successfully loaded FixedGoalBCNetwork (LatentDim={extracted_state_dim}) from {model_path}")
                    loaded_ok = True
                except Exception as e:
                    logging.error(f"Failed to load FixedGoalBCNetwork: {e}")
            elif "e2e" in os.path.basename(model_path):
                try:
                    from modules.spatial_model import E2EBCNetwork
                    test_policy = E2EBCNetwork(num_classes=6)
                    test_policy.to(self.device)
                    test_policy.load_state_dict(state_dict, strict=False)
                    loaded_policy = test_policy
                    self.model_name = os.path.basename(model_path)
                    logging.info(f"Successfully loaded E2EBCNetwork from {model_path}")
                    loaded_ok = True
                except Exception as e:
                    logging.error(f"Failed to load E2EBCNetwork: {e}")
                    
            if not loaded_ok:
                try:
                    extracted_inp_dim = state_dict['input_layer.weight'].shape[1]
                except Exception:
                    extracted_inp_dim = 128 if is_discrete else 32
                    
                for size in possible_sizes:
                    try:
                        # Try instantiating this size
                        from config import ACTION_DIM
                        test_policy = CQLNetwork(input_dim=extracted_inp_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, use_ln=not is_discrete, model_size=size)
                        test_policy.to(self.device)
                        
                        test_policy.load_state_dict(state_dict, strict=True)
                        
                        # If success:
                        loaded_policy = test_policy
                        self.model_name = os.path.basename(model_path)
                        logging.info(f"Successfully loaded {size.upper()} CQL Policy (Input={extracted_inp_dim}, Actions={ACTION_DIM}) from {model_path}")
                        loaded_ok = True
                        break
                        
                    except Exception as e:
                        logging.debug(f"Size {size} failed: {e}")
            
            if loaded_ok:
                self.policy = loaded_policy
                self.policy.eval()
                
                # [NEW] Globally generic Centroid loading for ANY configured policy
                if "_markov_control.pth" in model_path:
                    centroid_path = model_path.replace("_markov_control.pth", "_centroid.npy")
                elif "_oracle_control.pth" in model_path:
                    centroid_path = model_path.replace("_oracle_control.pth", "_centroid.npy")
                else:
                    centroid_path = model_path.replace("model", "centroid").replace(".pth", ".npy")
                
                if os.path.exists(centroid_path):
                    self.mu_goal = np.load(centroid_path)
                    logging.info(f"Successfully bound group-goal Centroid telemetry tracker from {os.path.basename(centroid_path)}")
                    
                self.goal_image_path = None
                goal_image_path = model_path.replace("model", "goal_image").replace(".pth", ".jpg")
                if os.path.exists(goal_image_path):
                    self.goal_image_path = goal_image_path
                    logging.info(f"Successfully bound nearest goal image visual wrapper from {os.path.basename(goal_image_path)}")
                
                # [NEW] For *_seek_cql models: compute centroid from goal directory + find nearest image
                if '_seek_cql' in os.path.basename(model_path) and self.mu_goal is None:
                    self._load_seek_goal_centroid(model_path)
                    
                return True
            else:
                logging.error(f"Could not fit weights at {model_path} to any standard model size.")
                return False

        except Exception as e:
            logging.error(f"Failed to load model {model_path}: {e}")
            return False

    def _load_seek_goal_centroid(self, model_path):
        """
        For *_seek_cql models: compute the goal centroid from the goal directory
        (e.g., data/sofa/ for sofa_seek_cql) using the CVE encoder, and find the
        closest image for the goal inset display.
        """
        import re, glob
        import torch
        from torchvision import transforms
        from PIL import Image
        
        # Extract goal name from model filename: "...-sofa_seek_cql_model.pth" -> "sofa"
        basename = os.path.basename(model_path)
        match = re.search(r'-(\w+)_seek_cql', basename)
        if not match:
            logging.warning(f"Could not extract goal name from seek model: {basename}")
            return
        
        goal_name = match.group(1)  # e.g., "sofa"
        data_dir = os.path.dirname(model_path)
        goal_dir = os.path.join(data_dir, goal_name)
        
        if not os.path.isdir(goal_dir):
            logging.warning(f"Goal directory not found: {goal_dir}")
            return
        
        goal_images = sorted(glob.glob(os.path.join(goal_dir, '*.jpg')))
        if not goal_images:
            logging.warning(f"No goal images found in {goal_dir}")
            return
        
        logging.info(f"Computing {goal_name} centroid from {len(goal_images)} images in {goal_dir}...")
        
        try:
            # Use the CVE encoder to encode all goal images
            encoder = getattr(self, 'encoder', None)
            if encoder is None:
                logging.warning("No CVE encoder available — cannot compute seek centroid")
                return
            
            device = next(encoder.parameters()).device
            img_dim = 64  # Standard for TinyVAE
            transform = transforms.Compose([
                transforms.Resize((img_dim, img_dim)),
                transforms.ToTensor(),
            ])
            
            all_latents = []
            batch_size = 32
            
            for i in range(0, len(goal_images), batch_size):
                batch_paths = goal_images[i:i+batch_size]
                tensors = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert('RGB')
                        tensors.append(transform(img))
                    except Exception:
                        continue
                if tensors:
                    batch_tensor = torch.stack(tensors).to(device)
                    with torch.no_grad():
                        z = encoder.encode(batch_tensor)
                    all_latents.append(z.cpu().numpy())
            
            if not all_latents:
                logging.error("Failed to encode any goal images")
                return
            
            latents = np.concatenate(all_latents, axis=0)
            centroid = latents.mean(axis=0)
            self.mu_goal = centroid
            logging.info(f"Computed {goal_name} centroid from {len(latents)} images (latent dim={centroid.shape[0]})")
            
            # Save centroid for future fast loading
            centroid_save_path = model_path.replace("model", "centroid").replace(".pth", ".npy")
            np.save(centroid_save_path, centroid)
            logging.info(f"Saved {goal_name} centroid to {os.path.basename(centroid_save_path)}")
            
            # Find the closest image to the centroid for the goal inset
            distances = np.linalg.norm(latents - centroid, axis=1)
            nearest_idx = int(np.argmin(distances))
            nearest_path = goal_images[nearest_idx]
            
            # Copy nearest image to the expected goal_image location
            goal_image_dest = model_path.replace("model", "goal_image").replace(".pth", ".jpg")
            import shutil
            shutil.copy2(nearest_path, goal_image_dest)
            self.goal_image_path = goal_image_dest
            logging.info(f"Bound nearest {goal_name} image as goal inset: {os.path.basename(nearest_path)} (dist={distances[nearest_idx]:.4f})")
            
        except Exception as e:
            logging.error(f"Failed to compute seek goal centroid: {e}")


    def _load_policy(self, specific_path=None):
        """
        Loads the Discrete CQL Policy (MLP).
        """
        logging.info("Loading CQLNetwork Policy...")
        # Input Dim defined in config (should be 64 for 32+32)
        policy = CQLNetwork(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=5)
        
        candidates = []
        candidates = []
        if specific_path:
            candidates.append(specific_path)
            
        # [MODIFIED] Disable Eager Loading of defaults
        # candidates.extend([
        #      os.path.join(MODELS_DIR, "cql_policy.pth"),
        #      os.path.join(os.path.dirname(MODELS_DIR), "data", "cql_policy.pth"),
        #      os.path.join(MODELS_DIR, "..", "data", "cql_policy.pth")
        # ])
        
        weights_loaded = False
        final_policy = policy # Default to small random

        if specific_path and "random_walk_baseline" in specific_path:
             logging.info("Intercepted Random Walk Baseline evaluation! Emitting dummy policy.")
             return RandomCQLPolicy()

        for path in candidates:
             if os.path.exists(path):
                 logging.info(f"Found policy weights at {path}")
                 try:
                     checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                     state_dict = checkpoint
                     
                     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                         state_dict = checkpoint['model_state_dict']
                         if 'valid_actions' in checkpoint:
                             self.valid_actions = checkpoint['valid_actions']
                         if 'hyperparameters' in checkpoint:
                             self.hyperparameters = checkpoint['hyperparameters']
                         # [NEW] Extract sculpted DrQ filters for the vision system
                         if 'encoder_state_dict' in checkpoint and hasattr(self, 'vision') and self.vision:
                             try:
                                 self.vision.model.load_state_dict(checkpoint['encoder_state_dict'])
                                 logging.info("Successfully bound end-to-end DrQ encoder filters.")
                             except Exception as e:
                                 logging.error(f"Failed to bind DrQ encoder during init load: {e}")

                     # [NEW] Dynamic Size Loading Loop
                     possible_sizes = ['tiny', 'small', 'medium', 'large', 'enormous', 'tectonic']
                     is_discrete = 'discrete' in os.path.basename(path).lower() and 'cve' not in os.path.basename(path).lower()
                     possible_input_dims = [512] if is_discrete else [32]
                     loaded_ok = False
                     
                     if "telemetry_oracle" in os.path.basename(path):
                         try:
                             from config import ACTION_DIM
                             from modules.spatial_model import OracleQNetwork
                             test_policy = OracleQNetwork(state_dim=4, goal_dim=4, action_dim=ACTION_DIM)
                             test_policy.to(self.device)
                             test_policy.load_state_dict(state_dict, strict=False)
                             final_policy = test_policy
                             self.model_name = os.path.basename(path)
                             logging.info(f"Successfully loaded OracleQNetwork (Input=8, Actions={ACTION_DIM}) from {path}")
                             loaded_ok = True
                         except Exception as e:
                             logging.error(f"Failed to load OracleQNetwork: {e}")
                             
                     if not loaded_ok:
                        try:
                            extracted_inp_dim = state_dict['input_layer.weight'].shape[1]
                        except Exception:
                            extracted_inp_dim = 128 if is_discrete else 32
                            
                        for size in possible_sizes:
                            try:
                                # Try instantiating this size
                                from config import ACTION_DIM
                                test_policy = CQLNetwork(input_dim=extracted_inp_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, use_ln=not is_discrete, model_size=size)
                                test_policy.to(self.device)
                                
                                # Strict Load
                                test_policy.load_state_dict(state_dict, strict=True)
                                
                                # If success:
                                final_policy = test_policy
                                self.model_name = os.path.basename(path)
                                logging.info(f"Successfully loaded {size.upper()} CQL Policy (Input={extracted_inp_dim}, Actions={ACTION_DIM}) from {path}")
                                loaded_ok = True
                                break
                                
                            except RuntimeError as e:
                                logging.debug(f"Size {size} failed: {e}")
                        if loaded_ok: break
                              
                     if loaded_ok:
                         weights_loaded = True
                         break
                     else:
                         logging.warning(f"Could not fit weights at {path} to any standard model size.")

                 except Exception as e:
                     logging.warning(f"Failed to load candidate {path}: {e}")
        
        if not weights_loaded:
             logging.info("No cql_policy.pth loaded. Using random weights (Small).")
             
        return final_policy


    def decide(self, z_current, state_vec=None, dist_threshold=None, continuous_z=None, img=None, distance_override=None):
        """
        Input: z_current (32 or 512,) Latent, state_vec (3,) Explicit State
               continuous_z: Optional 32-dim continuous embedding for distance calc (discrete VQ-VAE)
               img: Optional raw image frame for E2E models
        Output: (action_id, distance_to_goal, effective_threshold, goal_index, active_goal_dict, reflex_triggered)
        """
        now = time.time()
        
        if state_vec is None:
            # Fallback for old models expecting 3 dims
            state_vec = np.zeros(3, dtype=np.float32)
            
        # Ensure state dims are dynamic based on what's passed in
        state_dim = state_vec.shape[0]
        
        # 1. Latent Temporal Smoothing (EMA) disabled to mathematically match offline training
        alpha = 1.0 # Prev 0.3
        if self.z_smoothed is None:
            self.z_smoothed = z_current.copy()
        else:
            self.z_smoothed = alpha * z_current + (1.0 - alpha) * self.z_smoothed

        # [NEW] Check if dark_wall pure reflex model, group-goal, or classifier-seek model
        m_name = getattr(self, 'model_name', '')
        is_pure_reflex = m_name is not None and ('dark_wall' in m_name or 'dark-wall' in m_name or 'ir_reflex' in m_name)
        is_group_goal = m_name is not None and ('group-goal' in m_name or 'group_goal' in m_name or 'fixed_goal' in m_name or 'discrete_cql' in m_name)
        is_seek_model = m_name is not None and '_seek_cql' in m_name  # e.g. sofa_seek_cql, tv_seek_cql

        active_goal_dict = None
        goal_idx = -1
        z_goal = None
        
        # 2. Decide Active Goal (No cyclic patrol behavior)
        if not is_pure_reflex and not is_seek_model:
            if not self.goals and (not is_group_goal or getattr(self, 'mu_goal', None) is None):
                # Do not attempt to drive if no valid goal is set
                return 0, 10.0, dist_threshold if dist_threshold is not None else 0.0, 0, None, False
                
            if is_group_goal and getattr(self, 'mu_goal', None) is not None:
                z_goal = self.mu_goal
                active_goal_dict = {'latent': self.mu_goal} # Placeholder
                goal_idx = 0
            else:
                # Strictly use the structurally selected goal
                active_goal_dict = self.goals[self.current_goal_idx]
                z_goal = active_goal_dict['latent']
                
                # Ensure z_goal is a clean numpy array (handle raw tensors or nested arrays)
                if hasattr(z_goal, "cpu"):
                    z_goal = z_goal.cpu().detach().numpy()
                z_goal = np.array(z_goal).squeeze()
                goal_idx = self.current_goal_idx
        elif is_seek_model and getattr(self, 'mu_goal', None) is not None:
            z_goal = self.mu_goal
            active_goal_dict = {'latent': self.mu_goal}
            goal_idx = 0
        
        # 2b. Distance Check (Euclidean distance to the Central Goal Envelope)
        if not is_pure_reflex and z_goal is not None:
            # For discrete VQ-VAE: use continuous embeddings for meaningful distance
            continuous_goal = getattr(self, 'continuous_goal', None)
            if continuous_z is not None and continuous_goal is not None:
                # Both current and goal are in continuous pre-quantized space (32-dim)
                distance = distance_override if distance_override is not None else float(np.linalg.norm(continuous_z - continuous_goal))
            else:
                # Continuous VAE path: z_smoothed and z_goal are in the same space
                vision_dim = z_goal.shape[-1]
                vision_dist = self.z_smoothed[:vision_dim]
                distance = distance_override if distance_override is not None else float(np.linalg.norm(vision_dist - z_goal))
        else:
            distance = 10.0
        
        # 3. Policy Inference
        
        # [NEW] Handle End-To-End Image Policies
        from modules.spatial_model import E2EBCNetwork
        if isinstance(self.policy, E2EBCNetwork):
            if img is None:
                return 0, distance, dist_threshold if dist_threshold else 0.0, goal_idx, active_goal_dict, False
                
            # Perform torchvision ImageNet transforms
            import torchvision.transforms as transforms
            from PIL import Image
            import cv2
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
            img_t = transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.policy(img_t)
                action_id = torch.argmax(logits, dim=-1).item()
                
            return action_id, distance, dist_threshold if dist_threshold else 0.0, goal_idx, active_goal_dict, False

        # Assumption: Model input is ALWAYS 32 dims (1 fixed-goal frame)
        in_features = getattr(self.policy.input_layer, 'in_features', 0) if hasattr(self.policy, 'input_layer') else 0
        is_telemetry_oracle = in_features == 8
        
        if is_telemetry_oracle:
            obs = np.concatenate([self.z_smoothed, z_goal])
        else:
            obs = self.z_smoothed
        
        # Tensorize
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device) # (1, 128 or 96)
        
        with torch.no_grad():
            if m_name and "ir_reflex" in m_name:
                # Custom inference logic for IR Predictor
                # Evaluate all 3 valid actions (0=Forward, 1=Left, 2=Right)
                predicted_irs = []
                for act_idx in [0, 1, 2]:
                    act_tensor = torch.tensor(act_idx, dtype=torch.long, device=self.device)
                    ir_pred = self.policy(obs_tensor, act_tensor)
                    predicted_irs.append(ir_pred.item())
                
                # Pick action that MINIMIZES the predicted IR (smallest IR = further from wall)
                action = np.argmin(predicted_irs)
                current_q = -predicted_irs[action] # pseudo Q-value for logging
                raw_q = predicted_irs # We'll just display predicted IRs instead of Q values
            else:
                q_values = self.policy(obs_tensor)
                if q_values is None: return 0, distance, 0, active_goal_dict, False
                
                raw_q = q_values.cpu().numpy().tolist()[0]
                
                # Mask Invalid Actions
                if self.valid_actions:
                    mask = torch.full_like(q_values, float('-inf'))
                    for action_idx in self.valid_actions:
                        if action_idx < q_values.shape[1]:
                            mask[0, action_idx] = 0 # Unmask
                    
                    q_values = q_values + mask

                action = torch.argmax(q_values, dim=1).item()
                current_q = torch.max(q_values).item()
                
        # 3.5 [NEW] Map Dubins Output for Hello World Archetypes
        # Split ir_reflex (3 classes) from BC fixed_goal (4 classes)
        if m_name:
            if 'ir_reflex' in m_name or ('hello_world' in m_name and 'fixed_goal' not in m_name):
                dubins_map = {0: 1, 1: 3, 2: 4} # 0->FWD, 1->LEFT, 2->RIGHT
                action = dubins_map.get(action, 0)
            elif 'fixed_goal' in m_name and 'cql' not in m_name.lower():
                fixed_goal_map = {0: 1, 1: 2, 2: 3, 3: 4} # 0->FWD, 1->REV, 2->LEFT, 3->RIGHT
                action = fixed_goal_map.get(action, 0)
            
        # 4. Q-Value Saturation Logic
        delta_q = 0.0
        q_high_threshold = 25.0
        delta_q_threshold = 0.5
        q_saturation_triggered = False
        
        if self.last_q_value is not None:
            delta_q = current_q - self.last_q_value
            # If Q stops rising and is at a generally high terminal score, trigger arrival
            if abs(delta_q) < delta_q_threshold and current_q > q_high_threshold:
                q_saturation_triggered = True
                
        self.last_q_value = current_q
        
        # 5. Stop Logic (Reflex and Envelopes)
        from config import STOP_DISTANCE_THRESHOLD, STOP_COOLDOWN_S
        
        base_threshold = dist_threshold if dist_threshold is not None else STOP_DISTANCE_THRESHOLD
        
        # The arrival trigger operates strictly using the manual user threshold slider
        effective_threshold = base_threshold
             
        reflex_triggered = False
        if not is_pure_reflex:
            if distance <= effective_threshold:
                if m_name and ('fixed_goal' in m_name or 'discrete_cql' in m_name):
                    action = 5 # Intentional Stop
                    reflex_triggered = False
                else:
                    action = 0 # STOP is 0 per config/comms
                    reflex_triggered = True
                self.stop_cooldown_end_time = now + STOP_COOLDOWN_S
            elif hasattr(self, 'stop_cooldown_end_time') and now < self.stop_cooldown_end_time:
                if m_name and ('fixed_goal' in m_name or 'discrete_cql' in m_name):
                    action = 5
                    reflex_triggered = False
                else:
                    action = 0
                    reflex_triggered = True
            
        # Always log for debugging
        q_str = "[" + ", ".join([f"{x:.2f}" for x in raw_q]) + "]"
        # logging.info(f"DECISION: Act={action} | Dist={distance:.3f} | r={self.goal_radius:.2f} | dQ={delta_q:.2f} | maxQ={current_q:.2f} | Reflex={reflex_triggered}")
        return action, distance, effective_threshold, goal_idx, active_goal_dict, reflex_triggered
