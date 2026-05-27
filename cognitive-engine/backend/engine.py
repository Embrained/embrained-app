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


import time
import threading
import logging
import numpy as np
import cv2
import base64
import glob
import os
import asyncio

from backend.utils import safe_import_torch
torch = safe_import_torch()

from modules.comms import NervousSystem
from modules.vision import VisionSystem
from modules.planner import Planner
from modules.exploration import ExplorationSystem, MarkovWASD, AUTONOMY_THRESHOLD
from backend.manifold import ManifoldService
from backend.services.latent_slam_service import LatentSLAMService
try:
    from backend.server import AsyncPolicyServer
except ImportError:
    AsyncPolicyServer = None

from backend.controllers.dreamer_controller import DreamerController
# Game Modules
from game.referee import MotionReferee

# Configuration
from config import CONTROL_FREQ, GOAL_LED_COLORS, DATA_DIR, MODELS_DIR, GOAL_DIR, ACTION_NAMES, ACTION_PWM_MAP, COLOR_NAME_MAP, STOP_DISTANCE_THRESHOLD, RECORD_W, RECORD_H
from backend.vla import VLAController

# --- NEW ENGINE CORE ---
# --- NEW ENGINE CORE ---
from backend.engine_core.state import StateManager
from backend.engine_core.models import ModelManager
from backend.engine_core.commands import CommandDispatcher

class CognitiveEngine:
    def __init__(self, dry_run=False, robot_ip=None, stream_port=81, rec_prefix='markov', use_webcam=False, is_simulation=False):
        self.dry_run = dry_run
        self.is_simulation = is_simulation
        self.drive_mode = False 
        self.robot_ip = robot_ip
        self.stream_port = stream_port
        self.rec_prefix = rec_prefix
        self.use_webcam = use_webcam
        self.running = False
        self.thread = None
        self._stopped = False 
        
        # --- CORE COMPONENTS ---
        self.state_manager = StateManager()
        self.model_manager = ModelManager()
        self.dispatcher = CommandDispatcher(self)
        
        # Live Mode Persistent State
        self.current_live_action = 0 # Default STOP
        self.active_model_name = None 
        self.active_model_path = None
        self.stop_threshold = STOP_DISTANCE_THRESHOLD 
        self.reflex_enabled = True
        
        # Dynamic Objective Sampling
        self.runtime_transitions = []
        self.last_sampling_time = 0
        self.sampling_interval = 15.0  # seconds
        
        # Set Paths in State
        self.state_manager.update("data_root", DATA_DIR)
        
        # Command Queue 
        self.command_queue = []
        self.queue_lock = threading.Lock()
        
        # Recording State
        self.recording = False
        self.current_sound_cmd = "s:0;" 
        self.logger = None 

        # State Tracking 
        self.last_led_color = (0, 0, 0)
        self.last_sound_freq = 0

        # VLA State
        self.policy_server = None
        self.vla = None
        self.dreamer_ctrl = None
        self.slam_inference = None
        self.last_markov_state = 'WAITING' # [NEW] Track bout transitions

        # Game State
        self.referee = None
        self.game_mode = "GREEN_LIGHT" 
        

        
        # Initialize Modules
        logging.debug("Initializing Cognitive Engine...")
        
        has_cuda = torch.cuda.is_available() if torch else False
        self.device = 'cuda' if has_cuda else 'cpu'
        
        try:
            self.comms = None
            self.sim = None 
            
            # Auto-Discover Models via Manager
            # [MODIFIED] No Eager Loading - Start Clean
            # enc_path = self.model_manager.find_best_model("vae_encoder.pth")
            # if not enc_path:
            #     enc_path = self.model_manager.find_best_model("tiny_vae_final.pth")
            # 
            # pol_path = self.model_manager.find_best_model("cql_policy.pth")
            
            # Initialize with Defaults (Random Weights)
            logging.debug("Initializing VisionSystem...")
            self.vision = VisionSystem(device=self.device, model_path=None)
            logging.debug("VisionSystem Initialized.")

            # logging.debug("Refreshing Goal Latents...")
            # self._refresh_goal_latents()
            # logging.debug("Goal Latents Refreshed.")

            # Latent Space Components
            self.planner = Planner(device=self.device, model_path=None)
            self.explorer = ExplorationSystem() 
            self.cql_controller = MarkovWASD() # NEW CQL PACER
            self.manifold = ManifoldService(self.vision)
            logging.debug("Initializing LatentSLAMService...")
            self.latent_slam_service = LatentSLAMService()
            logging.debug("LatentSLAMService Initialized.")
            
            # VLA Init
            if AsyncPolicyServer:
                self.policy_server = AsyncPolicyServer()
            
            logging.debug("Initializing VLAController...")
            self.vla = VLAController(self.policy_server)
            logging.debug("VLAController Initialized.")
            logging.debug("Initializing MotionReferee...")
            self.referee = MotionReferee() 
            logging.debug("MotionReferee Initialized.") 
                
        except Exception as e:
            logging.critical(f"Engine Init Failed: {e}")
            raise e
            
        # UI should boot up without any goals preselected
        self.goal_imgs_b64 = []
        
        # [NEW] Alternating Fixed Goal Evaluation Feature
        self.fg_eval_phase = 'MODEL' # Alternates between 'MODEL' and 'MARKOV'
        self.fg_eval_bouts = 0
        self.fg_eval_reached = False
        self.fg_eval_step_count = float('nan')
        self.cql_eval_results = []
        self.latched_eval_dist = None
        self.latched_eval_action = 0
        
        # [NEW] Safety Reflex Trackers
        self.is_in_reflex = False
        self.last_reflex_end_time = 0.0

        # [NEW] Oracle Performance Tracking
        self.oracle_analytics = {
            'target_attempts': [],
            'current_steps': 0,
            'consecutive_stops': 0,
            'last_goal': None,
            'active': False
        }
        
        # Stats Tracking
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_frames": 0,
            "actions": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
            "action_names": ACTION_NAMES
        }
        
        # [NEW] Telemetry Extractor Proxy Init
        from scripts.extract_telemetry import TelemetryExtractor
        self.telemetry_extractor = None
        self.telemetry_init_frames = []
        self.telemetry_warmup_active = False
        self.telemetry_target_oracle = False
        self.telemetry_initialized = False
        
        # Resolve absolute path to software_suite for the cache
        suite_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        t_cache = os.path.join(suite_dir, "telemetry_cache.npz")
        
        if os.path.exists(t_cache):
            try:
                self.telemetry_extractor = TelemetryExtractor([])
                self.telemetry_extractor.load_cache(t_cache)
                logging.info(f"Loaded Live Telemetry Extraction Protocol.")
            except Exception as e:
                logging.error(f"Failed to load telemetry cache: {e}")
        
    # --- PROPERTIES FOR COMPATIBILITY ---
    @property
    def state(self):
        return self.state_manager.state
        
    @property
    def state_lock(self):
        return self.state_manager.lock

    # --- MODEL EVALUATION HELPERS ---
    def _print_cql_eval_results(self):
        if self.cql_eval_results:
            import math
            processed_results = [float('nan') if r == 1 else r for r in self.cql_eval_results]
            valid_results = [r for r in processed_results if not math.isnan(r)]
            total = len(processed_results)
            nan_count = total - len(valid_results)
            
            avg = sum(valid_results) / len(valid_results) if valid_results else 0
            nan_frac = (nan_count / total * 100) if total > 0 else 0
            
            # Format results gracefully including NaNs
            res_str = ", ".join(["NaN" if math.isnan(r) else str(r) for r in processed_results])
            
            params_str = ""
            if self.planner and hasattr(self.planner, 'hyperparameters'):
                hp = self.planner.hyperparameters
                if isinstance(hp, dict):
                    params_str = f" | Alpha={hp.get('cql_alpha', '?')}, Size={hp.get('model_size', '?').upper()}, Thr={hp.get('threshold_multiplier', '?')}x, Samp={hp.get('sampling_multiplier', '?'):.2f}x"
            
            # Identify name for native oracles
            eval_name = self.active_model_name
            if not eval_name and self.explorer and self.explorer.current_algo:
                eval_name = self.explorer.current_algo
            if not eval_name:
                eval_name = "Unknown Model"
            
            msg = f"[{eval_name}] Steps to reach goal (50-step max): {res_str} | Metric: {nan_frac:.1f}% NaNs, {avg:.1f} mean steps for successful.{params_str}"
            
            logging.info(msg)
            
            self.cql_eval_results = []


    # --- MODEL MANAGEMENT DELEGATES ---
    def _auto_adjust_threshold_for_model(self, model_name):
        if not model_name: return
        from config import STOP_DISTANCE_THRESHOLD
        # Distance is always computed in continuous embedding space (z_e),
        # so use the same threshold for both discrete and continuous architectures.
        if self.stop_threshold < 1.0:
            self.stop_threshold = STOP_DISTANCE_THRESHOLD
            logging.info(f"Auto-adjusted stop threshold to {self.stop_threshold} for {model_name}")

    def load_vae_model(self, model_filename):
        """Switches the Vision System VAE via ModelManager lookups."""
        logging.debug(f"Attempting to switch VAE to: {model_filename}")
        
        # [NEW] Check for Ground Truth Mode Before ModelManager Lookup
        if model_filename == "master_telemetry.csv":
            logging.info("Ground Truth Mode explicitly activated by user. Bypassing Neural Vision.")
            self.vision.enable_groundtruth(True)
            self.state_manager.update('bvae_model', "master_telemetry.csv")
            return True
        else:
            self.vision.enable_groundtruth(False)
        
        path = self.model_manager.find_best_model(model_filename)
        
        if not path:
             logging.error(f"Could not find VAE model: {model_filename}")
             return False
             
        logging.info(f"Loading VAE from: {path}")
        
        if self.vision.load_model(path):
             logging.debug("Reloading Manifold for new VAE...")
             if self.manifold:
                 self.manifold.set_model_name(model_filename, model_path=path)
                 self.manifold.is_ready = False
                 self.manifold.start_background_fit(force=False)
             # [NEW] Update State for UI
             self.state_manager.update('bvae_model', model_filename)
             self._auto_adjust_threshold_for_model(model_filename)
             self._refresh_goal_latents()
             return True
        else:
             logging.error("VisionSystem.load_model returned False")
        return False

    def load_cql_model(self, model_filename):
        """Helper to load CQL model, used by Dispatcher."""
        found_path = self.model_manager.find_best_model(model_filename)
        
        if found_path and os.path.exists(found_path):
            # [NEW] Auto-Switch VAE based on CQL name heuristic
            # Expected: "tinyvae-vae_20260226_113621-cql_20260226_120000.pth" -> "tinyvae-vae_20260226_113621.pth"
            try:
                # [NEW] Telemetry Oracle transition logic
                if "telemetry_oracle" in model_filename or "algorithmic_oracle" in model_filename:
                    logging.info("Telemetry Oracle engaged. Bypassing VAE load and enabling ground-truth mode.")
                    if hasattr(self, 'vision') and self.vision:
                        self.vision.groundtruth_mode = True
                        
                    # Bootstrapping Empty Room Tracker Physics via Manual Driver Pacing
                    # The telemetry initialization is now universally handled by the Telemetry Extension Hook
                    # at the end of the `_handle_set_controller` routine in commands.py.
                    pass
                else:
                    # 1. Parse the VAE base name from the CQL policy filename
                    if "-dark-wall-cql_" in model_filename:
                        vae_candidate = model_filename.split("-dark-wall-cql_")[0] + ".pth"
                    elif "-hello_world" in model_filename:
                        vae_candidate = model_filename.split("-hello_world")[0] + ".pth"
                    elif "-fixed_goal" in model_filename:
                        vae_candidate = model_filename.split("-fixed_goal")[0] + ".pth"
                    elif "-discrete_cql" in model_filename:
                        vae_candidate = model_filename.split("-discrete_cql")[0] + ".pth"
                    elif "-cql_" in model_filename:
                        vae_candidate = model_filename.split("-cql_")[0] + ".pth"
                    else:
                        # Fallback for old models
                        vae_candidate = model_filename.replace("-cql", "")
                        if not vae_candidate.endswith(".pth"):
                            vae_candidate += ".pth"
                            
                    # 3. Only switch if it's different from current
                    current_vae = self.state.get('bvae_model', '')
                    if vae_candidate != current_vae:
                         logging.debug(f"Auto-switching VAE to {vae_candidate} for policy {model_filename}")
                         success = self.load_vae_model(vae_candidate)
                         if not success:
                             logging.warning(f"Could not auto-load VAE {vae_candidate}. Manifold might be mismatched.")
                         if hasattr(self, 'vision') and self.vision:
                             self.vision.groundtruth_mode = False
            except Exception as e:
                logging.error(f"Error checking VAE for policy: {e}")

            if self.planner and self.planner.load_model(found_path):
                try:
                    checkpoint = torch.load(found_path, map_location=self.device, weights_only=True)
                    if isinstance(checkpoint, dict) and 'encoder_state_dict' in checkpoint:
                        if hasattr(self, 'vision') and self.vision and getattr(self.vision, 'encoder', None):
                            self.vision.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
                            logging.info("Successfully bound end-to-end DrQ encoder filters to runtime VisionSystem.")
                except Exception as e:
                    logging.error(f"Error checking for DrQ embedded filters: {e}")

                self._print_cql_eval_results()
                self.active_model_name = model_filename
                self.active_model_path = found_path 
                if self.explorer and not getattr(self, 'telemetry_target_oracle', False):
                    self.explorer.set_algorithm(None)
                
                # Clear conflicting controllers
                self.slam_inference = None
                self.dreamer_ctrl = None
                
                # [NEW] Do not preload goals on CQL load. Start clean until point is selected.
                self.planner.goals = []
                if hasattr(self.planner, 'latent_buffer'):
                    self.planner.latent_buffer.clear() # Clear frame stack on model switch
                self.planner.z_smoothed = None # [NEW] Drop old dimensions to prevent shape mismatch crash
                self.goal_imgs_b64 = []
                with self.state_lock:
                     self.state['goal_idx'] = 0
                     self.state['goal_image'] = None
                     self.state['goal_manifold_coords'] = []
                     self.state['goal_latents'] = []
                logging.info("Cleared goal slot and frame buffer for new CQL model. Awaiting manifold selection.")
                
                if getattr(self.planner, 'goal_image_path', None) and os.path.exists(self.planner.goal_image_path):
                     logging.info(f"Policy has embedded default goal image. Auto-loading: {self.planner.goal_image_path}")
                     self._update_runtime_goals([self.planner.goal_image_path], save_association=False)
                
                # We still want to load runtime transitions if they exist for dynamic sampling
                dataset_path = self.model_manager.infer_dataset_from_model(found_path, model_filename)
                
                if dataset_path:
                    try:
                        from backend.services.datasets import DatasetService
                        service = DatasetService(self.state_manager)
                        self.runtime_transitions = service.load_transitions(dataset_path)
                        logging.info(f"Loaded {len(self.runtime_transitions)} transitions for dynamic sampling.")
                        self.last_sampling_time = 0 # Force immediate sample
                    except Exception as e:
                        logging.error(f"Failed to load dataset transitions for sampling: {e}")
                        self.runtime_transitions = []
                else:
                     self.runtime_transitions = []
                self._auto_adjust_threshold_for_model(model_filename)
                return True
        return False

    def load_dreamer_model(self, model_filename):
        """Loads a DreamerV3 world model and policy."""
        logging.info(f"Attempting to load DreamerV3 Model: {model_filename}")
        found_path = self.model_manager.find_best_model(model_filename)
        
        if found_path and os.path.exists(found_path):
            try:
                self.dreamer_ctrl = DreamerController(found_path, device=self.device)
                self._print_cql_eval_results()
                self.active_model_name = model_filename
                self.active_model_path = found_path
                
                self.state_manager.update('controller', model_filename)
                
                if self.planner:
                    self.planner.load_model(None) # Disable VAE planner
                if self.explorer:
                    self.explorer.set_algorithm(None)
                
                # Clear conflicting controllers
                self.slam_inference = None
                
                # Refresh Manifold for Dreamer model if needed
                if self.manifold:
                    self.manifold.set_model_name(model_filename, model_path=found_path)
                    self.manifold.is_ready = False
                    self.manifold.start_background_fit(force=False)
                    
                logging.info(f"DreamerV3 Model loaded: {model_filename}")
                return True
            except Exception as e:
                logging.error(f"Failed to load DreamerV3 model {model_filename}: {e}")
        return False

    def load_slam_model(self, model_filename):
        """Loads a LatentSLAM GSSM model for live inference."""
        logging.info(f"Attempting to load LatentSLAM Model: {model_filename}")
        found_path = self.model_manager.find_best_model(model_filename)
        
        if found_path and os.path.exists(found_path):
            try:
                from backend.services.inference_service import LatentSLAMInference
                self.slam_inference = LatentSLAMInference(found_path, device=self.device)
                self._print_cql_eval_results()
                self.active_model_name = model_filename
                self.active_model_path = found_path
                self.state_manager.update('controller', model_filename)
                
                # Clear conflicting controllers
                self.dreamer_ctrl = None
                if self.planner:
                    self.planner.load_model(None) # Disable VAE planner
                if self.explorer:
                    self.explorer.set_algorithm(None)
                
                # Reset experience map for new model
                if self.latent_slam_service:
                    self.latent_slam_service.experience_map.nodes = []
                    self.latent_slam_service.experience_map.edges = []
                    
                # [NEW] Refresh Manifold for LatentSLAM model
                if self.manifold:
                    # Monkey patch for Manifold
                    class DummyVisionSLAM:
                        def __init__(self, slam):
                            self.device = slam.device
                            self.transform = slam.transform
                            class MockEncoder:
                                def __init__(self, m): self.m = m
                                def __call__(self, img):
                                    mu, logvar = self.m.encode(img)
                                    return None, mu, logvar
                                def eval(self): pass
                            self.encoder = MockEncoder(slam.model)
                    self.manifold.vision = DummyVisionSLAM(self.slam_inference)
                    
                    self.manifold.set_model_name(model_filename, model_path=found_path)
                    self.manifold.is_ready = False
                    self.manifold.start_background_fit(force=False)
                
                logging.info(f"LatentSLAM Inference loaded: {model_filename}")
                self._refresh_goal_latents()
                return True
            except Exception as e:
                logging.error(f"Failed to load SLAM model {model_filename}: {e}")
        return False

    def update_runtime_goals(self, image_paths, save_association=False):
        """Public wrapper for _update_runtime_goals to be called by Dispatcher."""
        return self._update_runtime_goals(image_paths, save_association)

    # --- ORIGINAL HELPER METHODS ---
    
    def _refresh_goal_latents(self):
        """Refreshes goal latents using the current Vision System."""
        
        # Determine the name of the active VAE model from state or filename
        vae_model_name = self.state.get("bvae_model", None)
        if vae_model_name == "N/A":
            vae_model_name = None
            
        if not vae_model_name and hasattr(self, 'active_model_path') and self.active_model_path:
            # Fallback if state not explicitly set but a path is active
            filename = os.path.basename(self.active_model_path)
            if "-dark-wall-cql_" in filename:
                vae_model_name = filename.split("-dark-wall-cql_")[0] + ".pth"
            elif "-hello_world" in filename:
                vae_model_name = filename.split("-hello_world")[0] + ".pth"
            elif "-fixed_goal" in filename:
                vae_model_name = filename.split("-fixed_goal")[0] + ".pth"
            elif "-discrete_cql" in filename:
                vae_model_name = filename.split("-discrete_cql")[0] + ".pth"
            elif "-cql_" in filename:
                vae_model_name = filename.split("-cql_")[0] + ".pth"
            else:
                vae_model_name = filename
                
        if vae_model_name and vae_model_name != "N/A":
            vae_base = vae_model_name.replace(".pth", "")
            goal_source_dir = os.path.join(DATA_DIR, f"{vae_base}_goals")
        else:
            goal_source_dir = os.path.join(DATA_DIR, "goals") # absolute fallback
            
        logging.info(f"Using goal source directory: {goal_source_dir}")

        if hasattr(self, 'active_model_path') and self.active_model_path:
            p = self.active_model_path
            # Try to infer dataset
            dataset_path = self.model_manager.infer_dataset_from_model(p)
            if dataset_path:
                candidate = os.path.join(dataset_path, f"{vae_base}_goals" if vae_model_name else "goals")
                if os.path.exists(candidate):
                    goal_source_dir = candidate
                    logging.info(f"Refreshing goals from Active Model context: {goal_source_dir}")

        if not os.path.exists(goal_source_dir):
            logging.debug(f"No goal directory found at {goal_source_dir}. Skipping encoding.")
            return

        goal_imgs = glob.glob(os.path.join(goal_source_dir, "*.jpg")) + glob.glob(os.path.join(goal_source_dir, "*.png"))
        goal_imgs.sort() 
        
        if not goal_imgs:
             logging.debug(f"No goal images found in {goal_source_dir}. Skipping encoding.")
             self.goal_imgs_b64 = [] 
             self.state_manager.update('goal_image', None)
             return

        self._update_runtime_goals(goal_imgs, save_association=False)

    def _load_goal_images_b64(self):
        input_dir = os.path.join(DATA_DIR, 'goals')
        if not os.path.exists(input_dir): return []
        files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + 
                       glob.glob(os.path.join(input_dir, "*.png")))
        images = []
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                b64 = base64.b64encode(buffer).decode('utf-8')
                images.append(b64)
        return images

    def _update_runtime_goals(self, image_paths, save_association=False):
        """Encodes goals and updates Planner + UI State. Bypasses encoding if latent is in cache."""
        if not image_paths:
            logging.warning("Empty goal list received.")
            if hasattr(self, 'planner'):
                self.planner.goals = []
                self.planner.current_goal_idx = 0
            
            self.goal_imgs_b64 = []
            self.state_manager.reset_goals_ui()
            return

        # Attempt to load precomputed 6-Channel JSON latents to bypass 3-channel JPEG limitations
        precomputed_goals = {}
        vae_model_name = self.state.get("bvae_model", None)
        if not vae_model_name and hasattr(self, 'active_model_path') and self.active_model_path:
            filename = os.path.basename(self.active_model_path)
            if "-dark-wall-cql_" in filename:
                vae_model_name = filename.split("-dark-wall-cql_")[0] + ".pth"
            elif "-hello_world" in filename:
                vae_model_name = filename.split("-hello_world")[0] + ".pth"
            elif "-fixed_goal" in filename:
                vae_model_name = filename.split("-fixed_goal")[0] + ".pth"
            elif "-discrete_cql" in filename:
                vae_model_name = filename.split("-discrete_cql")[0] + ".pth"
            elif "-cql_" in filename:
                vae_model_name = filename.split("-cql_")[0] + ".pth"
            else:
                vae_model_name = filename
                
        if vae_model_name and vae_model_name != "N/A":
            vae_base = vae_model_name.replace(".pth", "")
            json_path = os.path.join(DATA_DIR, f"{vae_base}_goals.json")
            if os.path.exists(json_path):
                import json
                try:
                    with open(json_path, 'r') as f:
                        goals_list = json.load(f)
                        for g in goals_list:
                            img_name = os.path.basename(g["image_path"])
                            precomputed_goals[img_name] = np.array(g["latent"], dtype=np.float32)
                    logging.info(f"Loaded {len(precomputed_goals)} precomputed goal latents from {json_path}")
                except Exception as e:
                    logging.error(f"Failed to load precomputed goal latents: {e}")

        new_latents = []
        new_slam_latents = []
        new_b64s = []
        new_coords = []
        new_rgbs = []
        
        try:
            for p_str in image_paths:
                full_path = p_str
                
                if not os.path.exists(full_path):
                    logging.warning(f"Goal image not found: {full_path}")
                    continue

                # 1. Load image and prepare Base64 string for UI
                img_cv = cv2.imread(full_path)
                if img_cv is None:
                    continue
                    
                # Save the raw RGB image for ViNT
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                new_rgbs.append(img_rgb)
                
                _, buffer = cv2.imencode('.jpg', img_cv)
                b64 = base64.b64encode(buffer).decode('utf-8')
                new_b64s.append(b64)

                # 2. Evaluate Latents (SLAM + VAE)
                slam_z = None
                if hasattr(self, 'slam_inference') and self.slam_inference:
                    try:
                        slam_z = self.slam_inference.encode_goal(img_cv)
                    except Exception as e:
                        logging.error(f"Failed to encode goal for LatentSLAM: {e}")
                        
                img_name = os.path.basename(full_path)
                
                # Extract Native Spatial Target for Telemetry Fallback
                t_goal = np.zeros(4, dtype=np.float32)
                try:
                    ts_str = img_name.replace('frame_', '').replace('webcam_frame_', '').replace('website_frame_', '').replace('.jpg', '')
                    import pandas as pd
                    import math
                    csv_path = os.path.join(DATA_DIR, "master_telemetry.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        ts_val = int(ts_str) if ts_str.isdigit() else ts_str
                        row = df[df['ts'] == ts_val]
                        if not row.empty:
                            r = row.iloc[0]
                            yaw_rad = math.radians(r['yaw_deg'])
                            t_goal = np.array([r['cx'] / 640.0, r['cy'] / 480.0, math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)
                except Exception as e:
                    pass
                
                if not hasattr(self, 'new_telemetry_goals'):
                    self.new_telemetry_goals = []
                self.new_telemetry_goals.append(t_goal)
                
                z = None
                if img_name in precomputed_goals:
                    z = precomputed_goals[img_name]
                elif getattr(self.vision, 'groundtruth_mode', False):
                    z = t_goal
                    if np.any(z):
                        logging.info(f"Loaded Native Spatial Target for Goal: {z}")
                    else:
                        logging.warning(f"Could not find timestamp {ts_str} in GT CSV. Defaulting to Zero-state.")
                else:
                    with open(full_path, 'rb') as f:
                        data = f.read()
                    _, z = self.vision.process_frame(data)
                
                # Capture continuous embedding for discrete VQ-VAE (used for distance + manifold)
                continuous_goal_z = getattr(self.vision, 'last_continuous_z', None)
                if continuous_goal_z is not None:
                    continuous_goal_z = continuous_goal_z.copy()
                
                # 3. Manifold Projection (Cache or Live)
                coord = None
                if hasattr(self, 'manifold') and self.manifold and self.manifold.is_ready:
                    pca_dim = getattr(self.manifold.pca, 'n_features_in_', 32) if hasattr(self.manifold, 'pca') else 32
                    
                    if full_path in self.manifold.library_paths and not (slam_z is not None and pca_dim > 64):
                        # Use purely cached point only if we aren't overriding it with a high-dim SLAM generation
                        idx = self.manifold.library_paths.index(full_path)
                        coord = self.manifold.manifold_points[idx]
                    else:
                        # Prefer continuous embedding for manifold projection (matches PCA dimensionality)
                        pca_dim = getattr(self.manifold.pca, 'n_features_in_', 32) if hasattr(self.manifold, 'pca') else 32
                        if continuous_goal_z is not None and continuous_goal_z.shape[0] == pca_dim:
                            target_z = continuous_goal_z
                        else:
                            target_z = slam_z if (slam_z is not None and pca_dim > 64) else z
                        if target_z is not None:
                            coord = self.manifold.project(target_z)
                            
                if z is not None:
                    new_latents.append(z)
                if coord is not None:
                    new_coords.append(coord)
                if slam_z is not None:
                    new_slam_latents.append(slam_z)
            
            # SLAM-only pipelines might not generate valid VAE latents, so we accept either structure as proof of goal processing
            if new_latents or new_slam_latents:
                num_processed = len(new_latents) if new_latents else len(new_slam_latents)
                logging.info(f"Successfully processed {num_processed} goals (cached & live).")
                
                if hasattr(self, 'planner'):
                    self.planner.goals = [{'latent': z} for z in new_latents]
                    for i, g in enumerate(new_rgbs):
                        if i < len(self.planner.goals):
                            self.planner.goals[i]['img'] = g
                    if hasattr(self.planner, 'latent_buffer'):
                        self.planner.latent_buffer.clear() # Clear framestack on new goal
                    self.planner.current_goal_idx = 0
                    self.planner.last_goal_switch = time.time()
                    self.cql_eval_new_goal_pending = True
                    
                    # Store continuous goal for distance computation (discrete VQ-VAE)
                    if continuous_goal_z is not None:
                        self.planner.continuous_goal = continuous_goal_z
                        logging.info(f"Stored continuous goal embedding (dim={continuous_goal_z.shape[0]}) for distance computation.")
                    
                self.goal_imgs_b64 = new_b64s
                
                if save_association and self.active_model_path and "-cql_" in os.path.basename(self.active_model_path):
                    self.model_manager.save_model_goals(self.active_model_path, image_paths)
                elif save_association:
                    logging.info(f"Skipped saving goals to {self.active_model_path} (Not a CQL policy file)")
                
                with self.state_lock:
                    self.state['goal_idx'] = 0
                    if new_b64s:
                        self.state['goal_image'] = new_b64s[0]
                    
                    self.state['goal_latents'] = [zl.tolist() if isinstance(zl, np.ndarray) else zl for zl in new_latents]
                    
                    if hasattr(self, 'new_telemetry_goals'):
                        self.state['telemetry_goal_coords'] = [g.tolist() if isinstance(g, np.ndarray) else g for g in self.new_telemetry_goals]
                        self.new_telemetry_goals = []
                    else:
                        self.state['telemetry_goal_coords'] = []
                        
                    if new_slam_latents:
                        self.state['goal_slam_latents'] = [sz.tolist() if isinstance(sz, np.ndarray) else sz for sz in new_slam_latents]
                    else:
                        self.state['goal_slam_latents'] = []
                    self.state['goal_manifold_coords'] = new_coords
            else:
                if (hasattr(self.vision, 'encoder') and self.vision.encoder) or (hasattr(self, 'slam_inference') and self.slam_inference):
                    logging.warning("No valid goals could be encoded from list.")
                else:
                    logging.debug("No models loaded to encode goals. Skipping encoding.")
                
        except Exception as e:
            logging.error(f"Failed to update runtime goals: {e}")



    # --- LIFECYCLE METHODS ---

    def start_feeds(self):
        if self.is_simulation:
            if not hasattr(self, 'sim') or self.sim is None:
                logging.info("Starting Native Virtual Simulation Engine (Raycast)...")
                try:
                    from modules.simulator import Simulator
                    self.sim = Simulator(headless=True, layout='room_rectangular')
                    with self.state_lock:
                        self.state['base_speed'] = 0.0
                        self.state['turn_speed'] = 0.0
                    logging.info("Raycast Simulator Started.")
                except Exception as e:
                    logging.error(f"Failed to start simulator: {e}")
                    self.sim = None
            return

        if hasattr(self, 'comms') and self.comms is not None:
            return 
            
        logging.debug("Starting Feeds (Camera/Serial)...")
        try:
            # [MODIFIED] Check for SpikerBot Quirk Mode
            use_quirks = (self.state.get('embodiment') == 'SPIKERBOT')
            self.comms = NervousSystem(
                dummy_mode=self.dry_run, 
                robot_ip=self.robot_ip, 
                stream_port=self.stream_port,
                spikerbot_quirks=use_quirks,
                use_webcam=self.use_webcam
            )
            logging.debug("Feeds Started.")
            # Initialize speed in state
            with self.state_lock:
                self.state['base_speed'] = self.comms.base_speed
                self.state['turn_speed'] = self.comms.turn_speed
        except Exception as e:
            logging.error(f"Failed to start feeds: {e}")
            self.comms = None

    def stop_feeds(self):
        if self.is_simulation:
            logging.info("Stopping Simulation Engine...")
            self.sim = None
            return

        if hasattr(self, 'comms') and self.comms is not None:
            logging.info("Stopping Feeds...")
            try:
                self.comms.send_command(0)
                self.comms.set_led((0,0,0))
                self.comms.send_sound_command(0)
                self.comms.close()
            except Exception as e:
                logging.error(f"Error stopping feeds: {e}")
            finally:
                self.comms = None
                logging.info("Feeds Stopped.")

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.stats['start_time'] = time.time()
        logging.debug("Cognitive Engine Started.")

    def set_mode(self, mode):
        if self.state.get('mode') == 'LIVE' and mode != 'LIVE':
             self._print_cql_eval_results()

        if mode == 'LIVE':
             self.start_feeds()
        else:
             self.stop_feeds()

        self.state_manager.set_mode(mode)
        if mode == 'LIVE':
             self.drive_mode = False 
             self.current_live_action = 0 
        elif mode == 'INFERENCE':
             self.drive_mode = True
            
        logging.debug(f"Mode switched to: {mode}")

    def handle_command(self, cmd_type, payload):
        """Delegate handling to Dispatcher via Queue."""
        with self.queue_lock:
            self.command_queue.append((cmd_type, payload))

    def set_manual_override(self, enabled):
        """Helper called by Dispatcher to switch off autonomy."""
        if enabled:
             self._print_cql_eval_results()
             if self.explorer:
                 self.explorer.set_algorithm(None)
             self.active_model_name = None

    def start_recording(self):
        if self.recording: return
        try:
            # Check if the 'Markov' or 'MarkovWASD' controller is currently active
            is_markov = (self.explorer and self.explorer.current_algo in ["Markov", "MarkovWASD"])
            
            from backend.services.markov_logger import MarkovLogger
            ctrl_name = self.explorer.current_algo if self.explorer and self.explorer.current_algo else "none"
            
            # Map MarkovTelemetry back to Markov for clean dataset pooling
            if getattr(self, 'telemetry_source_algo', None) == "MarkovTelemetry" or ctrl_name == "MarkovTelemetry":
                ctrl_name = "Markov"
                
            self.logger = MarkovLogger(controller_name=ctrl_name, prefix=self.rec_prefix)
                
            self.recording = True
            with self.state_lock:
                self.state['is_recording'] = True
        except Exception as e:
            logging.error(f"Failed to start recording: {e}")

    def stop_recording(self):
        if not self.recording: return
        self.recording = False
        with self.state_lock:
            self.state['is_recording'] = False
            self.state['recording_frames'] = 0
            
        if self.logger:
            self.logger.close()
            self.logger = None
        logging.info("Recording STOPPED.")
        self._dump_oracle_analytics()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if hasattr(self, 'comms') and self.comms is not None:
             # Only try to send stop if we have a valid, active connection
             try:
                 if hasattr(self.comms, 'ws') and self.comms.ws and self.comms.ws.connected:
                     self.comms.send_command(0) 
                     self.comms.set_led((0,0,0))
                     self.comms.send_sound_command(0)
             except Exception:
                 pass
             self.comms.close()
        
        self.stop_recording() 
        self._print_cql_eval_results()
        
        self.stats['end_time'] = time.time()
        self._print_summary()
        logging.debug("Cognitive Engine Stopped.")

    def _dump_oracle_analytics(self):
        if getattr(self, 'oracle_analytics', {}).get('active', False):
            self.oracle_analytics['active'] = False
            import numpy as np
            attempts = self.oracle_analytics['target_attempts']
            processed_attempts = [np.nan if a == 1 else a for a in attempts]
            valid_attempts = [a for a in processed_attempts if not np.isnan(a)]
            avg = np.mean(valid_attempts) if valid_attempts else 0
            med = np.median(valid_attempts) if valid_attempts else 0
            
            import logging
            logging.info("=========================================")
            logging.info("      --- ORACLE SESSION ENDED ---       ")
            logging.info(f"Target Attempts Log (Steps): {processed_attempts}")
            logging.info(f"Valid Inter-Goal Distances: {valid_attempts}")
            logging.info(f"Total Successful Reaches: {len(valid_attempts)}")
            if valid_attempts:
                logging.info(f"Average Steps to Target: {avg:.2f}")
                logging.info(f"Median Steps to Target: {med:.2f}")
                
            if hasattr(self, 'explorer') and hasattr(self.explorer, 'neural_oracle'):
                lats = getattr(self.explorer.neural_oracle, 'inference_latencies', [])
                if lats:
                    logging.info(f"Average MPC Inference Latency: {np.mean(lats):.3f}s (Max: {np.max(lats):.3f}s / Count: {len(lats)})")
                    self.explorer.neural_oracle.inference_latencies = []
            
            logging.info("=========================================")

    def _print_summary(self):
        duration = self.stats['end_time'] - self.stats['start_time']
        frames = self.stats['total_frames']
        fps = frames / duration if duration > 0 else 0
        
        # 1. Bin and Aggregate Actions
        binned = {}
        for action_id, count in self.stats['actions'].items():
            if isinstance(action_id, (tuple, list)):
                # Bin continuous velocities to nearest 0.1 for reporting
                bin_key = (round(float(action_id[0]), 1), round(float(action_id[1]), 1))
                binned[bin_key] = binned.get(bin_key, 0) + count
            else:
                binned[action_id] = binned.get(action_id, 0) + count

        # 2. Sort by frequency
        sorted_actions = sorted(binned.items(), key=lambda x: x[1], reverse=True)
        top_20 = sorted_actions[:20]

        summary = []
        summary.append("\n" + "="*40)
        summary.append("  EMBRAINED ENGINE SUMMARY STATS")
        summary.append("="*40)
        summary.append(f"  Duration      : {duration:.2f} seconds")
        summary.append(f"  Total Frames  : {frames}")
        summary.append(f"  Avg FPS       : {fps:.2f}")
        summary.append("-" * 40)
        summary.append(f"  ACTION DISTRIBUTION (Top {len(top_20)}):")
        
        for action_id, count in top_20:
            if isinstance(action_id, tuple):
                name = f"V={action_id[0]:.2f}, W={action_id[1]:.2f}"
            else:
                name = self.stats['action_names'].get(action_id, f"ID {action_id}")
            
            pct = (count / frames * 100) if frames > 0 else 0
            summary.append(f"    - {name:<15}: {count:<6} ({pct:.1f}%)")
        summary.append("="*40 + "\n")
        
        # Log as a single block to prevent interleaving
        logging.debug("\n".join(summary))

    def _reconnect_comms(self):
        logging.warning("Reconnecting communication system...")
        try:
             if hasattr(self, 'comms') and self.comms:
                  use_cam = self.comms.use_webcam
                  self.comms.close()
             else:
                  use_cam = self.use_webcam
        except Exception: 
             use_cam = False
        
        try:
             time.sleep(2.0)
             self.comms = NervousSystem(dummy_mode=self.dry_run, robot_ip=self.robot_ip, stream_port=self.stream_port)
             logging.info("Communication System Reconnected.")
        except Exception as e:
             logging.error(f"Reconnection Failed: {e}")

    def _sense(self):
        if self.is_simulation:
            if not hasattr(self, 'sim') or self.sim is None:
                return None, None, None
            frame = self.sim.get_latest_frame()
            webcam_frame = None
        else:
            if not hasattr(self, 'comms') or self.comms is None:
                 if self.dry_run or self.state['mode'] != 'LIVE':
                     img = np.zeros((RECORD_H, RECORD_W, 3), dtype=np.uint8)
                     return img, None, None
                 return None, None, None
                 
            frame = self.comms.get_latest_frame()
            webcam_frame = self.comms.get_latest_webcam_frame() if hasattr(self.comms, 'get_latest_webcam_frame') else None
        
        if frame is None and not self.dry_run:
            return None, None, None

        frame_to_process = frame
        if self.dry_run and frame is None:
             frame_to_process = np.zeros((RECORD_H, RECORD_W, 3), dtype=np.uint8)
        
        # [MODIFIED] Use LatentSLAM for z_cur if active
        z_cur = None
        img = None
        if hasattr(self, 'slam_inference') and self.slam_inference:
             # LatentSLAM gets the raw frame (BGR) and normalizes it.
             # We need last_action sent
             last_action = getattr(self.comms, 'last_action_pwm', (0,0)) if hasattr(self, 'comms') and self.comms else (0,0)
             z_cur = self.slam_inference.get_latent_state(frame_to_process, last_action)
             # VAE encoding for UI image (optional, we could just return the raw frame)
             img, _ = self.vision.process_frame(frame_to_process, webcam_input=webcam_frame)
        else:
             img, z_cur = self.vision.process_frame(frame_to_process, webcam_input=webcam_frame)
             
        return img, z_cur, webcam_frame

    def _extract_explicit_state(self, last_action_sent):
        curr_l, curr_r = 0.0, 0.0
        if last_action_sent is not None and len(last_action_sent) == 2:
             curr_l = float(last_action_sent[0])
             curr_r = float(last_action_sent[1])
             
        best_action = 0
        best_dist = float('inf')
        import math
        for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
            diff = math.hypot(curr_l - map_l, curr_r - map_r)
            if diff < best_dist:
                best_dist = diff
                best_action = act_id
                
        MAX_ACTION = 4.0
        action_norm = float(best_action) / MAX_ACTION
        
        curr_sonar = 0.0
        with self.state_lock:
             try: 
                 curr_sonar = float(self.state.get('sensor_dist', 0.0))
             except Exception: pass
                 
        dist_norm = curr_sonar / 1024.0
        state_vec = np.array([action_norm, dist_norm], dtype=np.float32)
        return state_vec, curr_sonar

    def _decide_cql_policy(self, z_cur, state_vec, curr_sonar):
        # We must call planner.decide every frame to maintain the VAE optical flow buffer (z_smoothed)
        # Pass continuous embedding for accurate distance computation with discrete architectures
        continuous_z = getattr(self.vision, 'last_continuous_z', None)
        action, dist, eff_thresh, goal_idx, active_goal_dict, reflex_triggered = self.planner.decide(
            z_cur, state_vec=state_vec, dist_threshold=self.stop_threshold, continuous_z=continuous_z
        )
        
        prev_state = getattr(self.cql_controller, 'state', None)
            
        target_action = 0
        is_bout_start = False
        
        if self.active_model_name and ('_markov_control' in self.active_model_name or '_oracle_control' in self.active_model_name):
            if reflex_triggered:
                action = 5
                reflex_triggered = False
                logging.info("Oracle Control hit Safety Net IR threshold. Triggering Intentional Stop.")
            elif action != 5:
                import random
                action = random.choices([1, 2, 3, 4], weights=[0.6, 0.1, 0.15, 0.15], k=1)[0]
                
        if getattr(self, 'fg_eval_reached', False):
            action = 5 # Force Intentional Stop during dwell phase
            reflex_triggered = False
        
        if reflex_triggered:
            self.cql_controller.state = 'WAITING'
            self.cql_controller.state_start_time = 0
            self.cql_controller.current_action_id = 0
            target_action = 0
        else:
            effective_sonar = curr_sonar if self.reflex_enabled else 0.0
            if self.active_model_name and ('fixed_goal' in self.active_model_name or 'discrete_cql' in self.active_model_name):
                effective_sonar = curr_sonar
                
            # [CRITICAL FIX] Prevent internal MarkovWASD safety reflexes from overriding Intentional Stop during Dwell
            if getattr(self, 'fg_eval_reached', False):
                effective_sonar = 0.0
                
            target_action = self.cql_controller.get_action(effective_sonar, teleop_action=action)
        
        new_state = getattr(self.cql_controller, 'state', None)
        if new_state == 'MOVE' and prev_state in ['STOP', 'WAITING']:
            is_bout_start = True
            
        return target_action, dist, goal_idx, reflex_triggered, is_bout_start

    def _decide_latent_slam(self, z_cur, curr_sonar):
        target_action = 0
        dist = 0.0
        with self.state_lock:
            goals = self.state.get('goal_slam_latents', [])
            if goals and len(goals) > 0:
                z_goal = np.array(goals[self.state.get('goal_idx', 0)]).squeeze()
                latent_dim = self.slam_inference.model.latent_dim
                if z_goal.shape[0] > latent_dim:
                    z_goal = z_goal[:latent_dim]
                    
                z_cur_slam = z_cur.squeeze()
                if z_cur_slam.shape[0] > latent_dim:
                    z_cur_slam = z_cur_slam[:latent_dim]

                alpha = 0.3
                if not hasattr(self, 'slam_z_smoothed') or self.slam_z_smoothed is None:
                    self.slam_z_smoothed = z_cur_slam.copy()
                else:
                    self.slam_z_smoothed = alpha * z_cur_slam + (1.0 - alpha) * self.slam_z_smoothed

                dist = float(np.linalg.norm(self.slam_z_smoothed - z_goal))
                
                if dist < self.stop_threshold:
                    action = 0
                else:
                    best_action = 0
                    min_dist = float('inf')
                    valid_actions = [0, 1, 3, 4]
                    
                    for act_id in valid_actions:
                        pwm = ACTION_PWM_MAP.get(act_id, (0,0))
                        z_next = self.slam_inference.predict_next_state(pwm)
                        z_next_sq = z_next.squeeze()
                        if z_next_sq.shape[0] > latent_dim:
                            z_next_sq = z_next_sq[:latent_dim]
                            
                        d = float(np.linalg.norm(z_next_sq - z_goal))
                        if d < min_dist:
                            min_dist = d
                            best_action = act_id
                            
                    action = best_action
                target_action = self.cql_controller.get_action(curr_sonar, teleop_action=action)
        return target_action, dist

    def _decide_exploration(self, current_mode, z_cur, img, dist, telemetry_cur=None):
        target_action = 0
        is_bout_start = False
        
        if getattr(self, 'dreamer_ctrl', None) and current_mode == 'INFERENCE':
            target_action = self.dreamer_ctrl.get_action(img=img, latent=z_cur)
        elif self.policy_server and current_mode == 'INFERENCE':
            target_action = self.vla.get_action(img)
        elif self.explorer and self.explorer.current_algo:
            try:
                s_dist = float(self.state.get('sensor_dist', '999'))
            except Exception:
                s_dist = 999.0
            teleop_val = self.current_live_action
            if getattr(self, 'fg_eval_reached', False):
                teleop_val = 5 # Force Intentional Stop during dwell phase
            
            algo = self.explorer.current_algo
            if algo == "Markov": ctrl = self.explorer.markov
            elif algo == "MarkovTelemetry": ctrl = getattr(self.explorer, 'markov_telemetry', None)
            elif algo == "MarkovWASD": ctrl = getattr(self.explorer, 'markov_wasd', None)
            elif algo == "MarkovSweep": ctrl = getattr(self.explorer, 'markov_sweep', None)
            elif algo == "Algorithmic Oracle": ctrl = getattr(self.explorer, 'algo_oracle', None)
            elif isinstance(algo, str) and algo.startswith("Neural Oracle"): ctrl = getattr(self.explorer, 'neural_oracle', None)
            else: ctrl = None
            
            prev_state = getattr(ctrl, 'state', None) if ctrl else None
            effector_dist = s_dist if self.reflex_enabled else 0.0
            
            # [CRITICAL FIX] Prevent internal MarkovWASD safety reflexes from overriding Intentional Stop during Dwell
            if getattr(self, 'fg_eval_reached', False):
                effector_dist = 0.0
            
            current_goal = None
            current_img_goal = None
            if hasattr(self, 'planner') and self.planner and self.planner.goals:
                if 0 <= self.planner.current_goal_idx < len(self.planner.goals):
                    current_goal = self.planner.goals[self.planner.current_goal_idx].get('latent', None)
                    current_img_goal = self.planner.goals[self.planner.current_goal_idx].get('img', None)
                    
            telemetry_goal = None
            if current_goal is None and z_cur is not None:
                with self.state_lock:
                    goals = self.state.get('goal_latents', [])
                    idx = self.state.get('goal_idx', 0)
                    if goals and len(goals) > 0 and 0 <= idx < len(goals):
                        current_goal = np.array(goals[idx]).squeeze()
                    
                    t_goals = self.state.get('telemetry_goal_coords', [])
                    if t_goals and len(t_goals) > 0 and 0 <= idx < len(t_goals):
                        telemetry_goal = np.array(t_goals[idx]).squeeze()
                        
            target_action = self.explorer.get_action(
                sensor_dist=effector_dist, teleop_action=teleop_val, 
                z_cur=z_cur, z_goal=current_goal, 
                img_cur=img, img_goal=current_img_goal, latent_dist=dist,
                telemetry_cur=telemetry_cur, telemetry_goal=telemetry_goal
            )
            
            new_state = getattr(ctrl, 'state', None) if ctrl else None
            if new_state == 'MOVE' and prev_state in ['STOP', 'WAITING']:
                is_bout_start = True
        else:
            target_action = self.current_live_action
            
        return target_action, is_bout_start

    def _project_manifold(self, z_cur):
        if self.manifold:
            latent_to_project = None
            # Prefer continuous embedding for discrete VQ-VAE architectures
            # (one-hot z_cur would fail the PCA dimension check)
            continuous_z = getattr(self.vision, 'last_continuous_z', None)
            if continuous_z is not None:
                latent_to_project = continuous_z
            elif z_cur is not None:
                latent_to_project = z_cur
            elif getattr(self, 'dreamer_ctrl', None) and self.dreamer_ctrl.last_latent is not None:
                latent_to_project = self.dreamer_ctrl.last_latent
            
            if latent_to_project is not None:
                pca_dim = getattr(self.manifold.pca, 'n_features_in_', 32) if hasattr(self.manifold, 'pca') else 32
                if isinstance(latent_to_project, np.ndarray) and latent_to_project.shape[0] > pca_dim:
                    latent_to_project = latent_to_project[:pca_dim]
                    
                coords = self.manifold.project(latent_to_project)
                with self.state_lock:
                    self.state['manifold_coord'] = coords if coords else None
            else:
                with self.state_lock:
                    self.state['manifold_coord'] = None
                    
            m_name = getattr(self.planner, 'model_name', '') if hasattr(self, 'planner') else ''
            fallback_latent = None
            if m_name and ('group-goal' in m_name or 'group_goal' in m_name or 'fixed_goal' in m_name or 'discrete_cql' in m_name):
                # Prefer continuous goal embedding for discrete architectures (matches PCA dim)
                if hasattr(self, 'planner'):
                    continuous_goal = getattr(self.planner, 'continuous_goal', None)
                    if continuous_goal is not None:
                        fallback_latent = continuous_goal
                    else:
                        fallback_latent = getattr(self.planner, 'mu_goal', None)
            elif getattr(self, 'explorer', None) and self.explorer.current_algo == "Neural Oracle":
                n_oracle = getattr(self.explorer, 'neural_oracle', None)
                if n_oracle: fallback_latent = getattr(n_oracle, 'goal_latent', None)
                
            if fallback_latent is not None:
                with self.state_lock:
                    if not self.state.get('goal_manifold_coords'):
                        if hasattr(fallback_latent, 'detach'):
                            fallback_latent = fallback_latent.detach().cpu().numpy().squeeze()
                        c_coord = self.manifold.project(fallback_latent)
                        if c_coord:
                            self.state['goal_manifold_coords'] = [c_coord]
                            self.state['goal_idx'] = 0

    def _decide(self, current_mode, z_cur, img=None, last_action_sent=(0, 0), telemetry_cur=None):
        target_action = 0 
        dist = 0.0
        goal_idx = 0
        reflex_triggered = False
        is_bout_start = False
        
        state_vec, curr_sonar = self._extract_explicit_state(last_action_sent)

        is_latentslam_active = hasattr(self, 'slam_inference') and self.slam_inference is not None and self.active_model_name and 'latentslam' in self.active_model_name.lower()
        
        if self.active_model_name and self.planner and not is_latentslam_active and not getattr(self, 'telemetry_warmup_active', False):
             if z_cur is not None:
                 target_action, dist, goal_idx, reflex_triggered, is_bout_start = self._decide_cql_policy(z_cur, state_vec, curr_sonar)
        elif is_latentslam_active:
             if z_cur is not None:
                 target_action, dist = self._decide_latent_slam(z_cur, curr_sonar)
        else:
             # Compute continuous distance even in teleop/exploration mode
             continuous_z = getattr(self.vision, 'last_continuous_z', None) if hasattr(self, 'vision') else None
             continuous_goal = getattr(self.planner, 'continuous_goal', None) if hasattr(self, 'planner') else None
             if continuous_z is not None and continuous_goal is not None:
                 dist = float(np.linalg.norm(continuous_z - continuous_goal))
             elif z_cur is not None:
                 with self.state_lock:
                     goals = self.state.get('goal_latents', [])
                     if goals and len(goals) > 0:
                         z_goal_tmp = np.array(goals[self.state.get('goal_idx', 0)]).squeeze()
                         try:
                             dist = float(np.linalg.norm(z_cur.squeeze() - z_goal_tmp))
                         except ValueError:
                             pass
              
             target_action, is_bout_start = self._decide_exploration(current_mode, z_cur, img, dist, telemetry_cur=telemetry_cur)

        if self.reflex_enabled and curr_sonar > AUTONOMY_THRESHOLD:
             reflex_triggered = True

        self._project_manifold(z_cur)

        active_ctrl = self.active_model_name if self.active_model_name else (self.explorer.current_algo if self.explorer else None)
        
        # Use live distance every frame (no latching)
        final_dist = dist

        with self.state_lock:
             self.state['controller'] = active_ctrl
             self.state['fg_eval_phase'] = getattr(self, 'fg_eval_phase', 'MODEL')
             self.state['latent_dist'] = final_dist
             self.state['latent_thresh'] = self.stop_threshold
             
        return target_action, final_dist, goal_idx, reflex_triggered, is_bout_start
    def _act(self, current_mode, target_action, last_action_sent, last_sent_time):
        now = time.time()
        should_drive = (current_mode == 'INFERENCE' and self.drive_mode) or (current_mode == 'LIVE')
        
        # 1. Convert Target Action to Raw PWM Motors
        if isinstance(target_action, int):
            # Discrete Manual Control (WASD) or Autonomous Policy
            if (hasattr(self, 'comms') and self.comms) or self.is_simulation:
                # [LIVE and INFERENCE] Directly use the perfect rigid 5-class PWM map
                # Map the Model's Action ID directly to the Explicit Target PWMs
                mapped_pwm = ACTION_PWM_MAP.get(target_action, (0, 0))
                effective_action = mapped_pwm
                
                if current_mode == 'LIVE' and not self.is_simulation:
                    # Update internal tracking for UI
                    if hasattr(self.comms, 'manual_v_pwm'):
                        self.comms.manual_v_pwm = mapped_pwm[0]
                        self.comms.manual_w_pwm = mapped_pwm[1]
                        self.comms.base_speed = abs(mapped_pwm[0])
                        self.comms.turn_speed = abs(mapped_pwm[1])
                        self.comms.last_action_id = target_action
            else:
                effective_action = ACTION_PWM_MAP.get(target_action, (0, 0)) # Fallback mapping
        else:
            # Continuous Control (Model/Wandering)
            v_raw, w_raw = target_action if (should_drive and isinstance(target_action, (tuple, list, np.ndarray))) else (0.0, 0.0)
            if abs(v_raw) > 1.1 or abs(w_raw) > 1.1:
                # Already raw PWM integers
                effective_action = (int(v_raw), int(w_raw))
            else:
                # Normalized floats -> Remap to PWM integers
                if hasattr(self, 'comms') and self.comms and self.comms.robot:
                    l_tgt, r_tgt = self.comms.robot.get_motor_commands(v_raw, w_raw)
                    effective_action = (l_tgt, r_tgt)
                else:
                    effective_action = (0, 0)

        # Force Stop if not driving
        if not should_drive:
            effective_action = (0, 0)
            
        # [REMOVED] Centralized Safety Reflex: IR initiation logic has been handed down to Exploration controllers directly.
        
        # 2. Decision to Send (Change OR Heartbeat)
        def active_change(a, b):
            if isinstance(a, tuple) and isinstance(b, tuple):
                return a[0] != b[0] or a[1] != b[1]
            return a != b

        # [NEW] Restore Heartbeat (100ms) and ensure UI reflects target even when not sending
        should_send = (current_mode in ['LIVE', 'INFERENCE']) and \
                      (active_change(effective_action, last_action_sent) or (now - last_sent_time > 0.1))
        
        motor_cmd_str = f"l:{int(effective_action[0])};r:{int(effective_action[1])};"
        new_last_action = last_action_sent
        new_last_sent_time = last_sent_time

        if self.is_simulation and hasattr(self, 'sim') and self.sim:
            if should_send:
                sim_action_id = 0
                if isinstance(target_action, int):
                    sim_action_id = target_action
                else:
                    for a_id, pwm_val in ACTION_PWM_MAP.items():
                        if pwm_val == effective_action and a_id != 5:
                            sim_action_id = a_id
                            break
                            
                self.sim.send_command(sim_action_id)
                motor_cmd_str = f"l:{int(effective_action[0])};r:{int(effective_action[1])};"
                new_last_action = effective_action
                new_last_sent_time = now
        elif should_send and hasattr(self, 'comms') and self.comms:
            # Send raw PWM integers
            res = self.comms.send_pwm(*effective_action)
            
            if res is None:
                raise ConnectionResetError("Motor command sending failed")
            
            motor_cmd_str = res
            new_last_action = effective_action
            new_last_sent_time = now

        return effective_action, motor_cmd_str, new_last_action, new_last_sent_time
    
    def _update_state(self, img, z_cur, target_action, effective_action, dist, goal_idx, last_goal_idx, motor_cmd_str, **kwargs):
        self.stats['total_frames'] += 1
        self.stats['actions'][effective_action] = self.stats['actions'].get(effective_action, 0) + 1
        
        # [NEW] Real-time LatentSLAM Inference & Marker
        if hasattr(self, 'slam_inference') and self.slam_inference:
            try:
                # Run inference on unprocessed frame (if available) or current img
                # Note: slam_inference expects BGR
                latent_mu = self.slam_inference.get_latent_state(img, effective_action)
                
                if self.latent_slam_service:
                    # Update the live experience map
                    ghost_mu = self.slam_inference.predict_next_state(effective_action)
                    self.latent_slam_service.update_live_state(latent_mu, prior_mu=ghost_mu)
            except Exception as e:
                logging.error(f"SLAM inference failed in update_state: {e}")

        with self.state_lock:
             ui_img = kwargs.get('ui_img', img)
             if ui_img is not None:
                 _, buffer = cv2.imencode('.jpg', ui_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                 self.state['image'] = base64.b64encode(buffer).decode('utf-8')
             self.state['use_webcam'] = getattr(self, 'use_webcam', False)
             
             webcam_ui_img = kwargs.get('webcam_ui_img', None)
             if webcam_ui_img is not None:
                 webcam_ui_img = cv2.resize(webcam_ui_img, (0, 0), fx=0.5, fy=0.5)
                 _, w_buffer = cv2.imencode('.jpg', webcam_ui_img, [cv2.IMWRITE_JPEG_QUALITY, 40])
                 self.state['webcam_image'] = base64.b64encode(w_buffer).decode('utf-8')
             else:
                 self.state['webcam_image'] = None
             
             led_cmd_str = "d:N/A;"
             new_last_goal_idx = last_goal_idx
             
             if goal_idx != last_goal_idx:
                 new_last_goal_idx = goal_idx
                 if GOAL_LED_COLORS and goal_idx < len(GOAL_LED_COLORS):
                     color = GOAL_LED_COLORS[goal_idx]
                     if color != self.last_led_color and hasattr(self, 'comms') and self.comms:
                         led_cmd_str = self.comms.set_led(color)
                         self.last_led_color = color
            
             if self.recording and self.logger:
                 from backend.services.markov_logger import MarkovLogger
                 if isinstance(self.logger, MarkovLogger):
                     # [NEW] SMDP Snapshot Logic for MarkovLogger                     # Standard execution triggers discrete logging snapshots
                     # IMPORTANT: motor_cmd_str might be stale or generated before comms responds.
                     # Generate an explicit string from effective_action to ensure correct logging.
                     if kwargs.get('is_bout_start', False):
                         explicit_cmd_str = f"l:{int(effective_action[0])};r:{int(effective_action[1])};"
                         
                         webcam_frame = None
                         if self.use_webcam and self.comms and hasattr(self.comms, 'get_latest_webcam_frame'):
                             webcam_frame = self.comms.get_latest_webcam_frame()
                             
                         self.logger.log_step(
                             frame=img,
                             ir_raw=self.comms.telemetry.get('dist', '0') if (self.comms and hasattr(self.comms, 'telemetry')) else '0',
                             batt_raw=self.comms.telemetry.get('batt', '0') if (self.comms and hasattr(self.comms, 'telemetry')) else '0',
                             ping_raw=self.comms.telemetry.get('ping', '0') if (self.comms and hasattr(self.comms, 'telemetry')) else '0',
                             motor_str=explicit_cmd_str,
                             active_controller=kwargs.get('active_controller_str', 'unknown'),
                             webcam_frame=webcam_frame
                         )
                         

                     self.state['recording_frames'] = self.logger.frame_count
             else:
                 self.state['recording_frames'] = 0
                 
             # [NEW] Live Moving Telemetry Warmup Buffer (Independent of recording state)
             webcam_frame_warmup = None
             if self.use_webcam and hasattr(self, 'comms') and self.comms and hasattr(self.comms, 'get_latest_webcam_frame'):
                 webcam_frame_warmup = self.comms.get_latest_webcam_frame()
                 
             if kwargs.get('is_bout_start', False) and self.telemetry_warmup_active and webcam_frame_warmup is not None and getattr(self, 'telemetry_extractor', None) is not None:
                 gray = cv2.cvtColor(webcam_frame_warmup, cv2.COLOR_BGR2GRAY)
                 self.telemetry_init_frames.append(gray)
                 logging.info(f"Telemetry Warmup Marker: {len(self.telemetry_init_frames)}/10 frames captured.")
                 
                 if len(self.telemetry_init_frames) >= 10:
                     # 1. Initialize
                     logging.info("10 distinct states gathered. Initializing moving telemetry struct...")
                     self.telemetry_extractor.initialize_moving_background(self.telemetry_init_frames)
                     self.telemetry_warmup_active = False # Telemetry plotting dynamically enabled!
                     self.telemetry_initialized = True
                     
                     # 2. Oracle Handoff Sequence
                     if getattr(self, 'telemetry_target_oracle', False):
                         logging.info("Oracle Telemetry Target configured. Executing Hot-Swap out of MarkovWASD into Oracle Engine!")
                         self.telemetry_target_oracle = False
                         source_algo = getattr(self, 'telemetry_source_algo', None)
                         
                         if hasattr(self, 'explorer'):
                             if source_algo == "Algorithmic Oracle":
                                 logging.info(f"Returning control to {source_algo}")
                                 self.explorer.set_algorithm(source_algo)
                             else:
                                 self.explorer.set_algorithm(None)
                             
                         if hasattr(self, 'planner') and self.planner:
                             # Grab the latest coords file dynamically
                             import glob
                             coords_files = glob.glob(os.path.join(DATA_DIR, "*_oracle_coords.npy"))
                             if coords_files:
                                 coords_files.sort(key=os.path.getmtime, reverse=True)
                                 target_path = coords_files[0]
                             else:
                                 target_path = ''
                                 
                             if self.active_model_path:
                                 candidate_path = self.active_model_path.replace('_oracle_control.pth', '_oracle_coords.npy')
                                 if os.path.exists(candidate_path): target_path = candidate_path
                                 
                             if os.path.exists(target_path):
                                 try:
                                     coords = np.load(target_path)
                                     self.planner.goals = [{'latent': coords, 'image': None}]
                                     logging.info(f"Loaded specific Oracle Telemetry Goal: {coords}")
                                 except Exception:
                                     self.planner.goals = [{'latent': np.array([0.5, 0.5, 1.0, 0.0], dtype=np.float32), 'image': None}]
                             else:
                                 self.planner.goals = [{'latent': np.array([0.5, 0.5, 1.0, 0.0], dtype=np.float32), 'image': None}]
                                 logging.info("Injected explicit default Arena Center goal: [0.5, 0.5, 1.0, 0.0]")
                             self.planner.current_goal_idx = 0
                             
                         if source_algo != "Algorithmic Oracle":
                             self.state['mode'] = 'INFERENCE' # Free inference loop from live bounds!
                     else:
                         source_algo = getattr(self, 'telemetry_source_algo', None)
                         if hasattr(self, 'explorer') and source_algo is not None:
                             logging.info(f"Warmup Complete. Transferring live autonomy to {source_algo}")
                             self.explorer.set_algorithm(source_algo)
                         self.telemetry_source_algo = None
             if 0 <= goal_idx < len(self.goal_imgs_b64):
                 self.state['goal_image'] = self.goal_imgs_b64[goal_idx]
             else:
                 self.state['goal_image'] = None
                 
             semantic_action = kwargs.get('semantic_action', target_action)
             if isinstance(semantic_action, tuple):
                 # Pluck the current integer action ID out of the active pacer or explorer
                 if getattr(self, 'fg_eval_phase', 'MODEL') == 'MARKOV' and hasattr(self, 'explorer') and hasattr(self.explorer, 'markov'):
                     markov_action = getattr(self.explorer.markov, 'current_action', 'STOP')
                     if getattr(self.explorer.markov, 'state', 'STOP') == 'STOP': markov_action = 'STOP'
                     map_dict = {'FWD': 1, 'BACK': 2, 'LEFT': 3, 'RIGHT': 4, 'STOP': 0}
                     semantic_action = map_dict.get(markov_action, 0)
                 else:
                     current_algo = getattr(self.explorer, 'current_algo', None) if getattr(self, 'explorer', None) else None
                     if current_algo == "Algorithmic Oracle" and getattr(self.explorer, 'algo_oracle', None) and hasattr(self.explorer.algo_oracle.pacer, 'current_action_id'):
                         semantic_action = getattr(self.explorer.algo_oracle.pacer, 'current_action_id', 0)
                     elif hasattr(self, 'cql_controller') and self.cql_controller and hasattr(self.cql_controller, 'current_action_id'):
                         semantic_action = getattr(self.cql_controller, 'current_action_id', 0)
                     elif hasattr(self, 'explorer') and self.explorer and hasattr(self.explorer, 'current_action_id'):
                         semantic_action = getattr(self.explorer, 'current_action_id', 0)

             action_label = ACTION_NAMES.get(semantic_action, '?')
             if semantic_action == 0 and 'reflex_triggered' in kwargs and kwargs['reflex_triggered']:
                 action_label = "STOP (ARRIVAL REFLEX)"
             elif semantic_action == 2 and 'reflex_triggered' in kwargs and kwargs['reflex_triggered']:
                 action_label = "REVERSE (OBSTACLE REFLEX)"
             
             self.state['action'] = action_label
             self.state['distance'] = float(dist)
             self.state['goal_idx'] = goal_idx
             self.state['reflex_enabled'] = self.reflex_enabled
             
             # Export evaluation state for UI
             self.state['telemetry_warmup_active'] = getattr(self, 'telemetry_warmup_active', False)
             self.state['telemetry_init_frames_left'] = max(0, 10 - len(getattr(self, 'telemetry_init_frames', [])))
             self.state['fg_eval_reached'] = getattr(self, 'fg_eval_reached', False)
             if getattr(self, 'fg_eval_reached', False):
                 self.state['fg_eval_dwell_left'] = max(0, (getattr(self, 'fg_eval_step_count', 0) + 10) - getattr(self, 'fg_eval_bouts', 0))
             else:
                 self.state['fg_eval_dwell_left'] = 0
             self.state['fg_eval_timeout_left'] = max(0, 50 - getattr(self, 'fg_eval_bouts', 0))
             self.state['fg_eval_bouts'] = getattr(self, 'fg_eval_bouts', 0)
             self.state['telemetry_source_algo'] = getattr(self, 'telemetry_source_algo', None)
             self.state['active_model_name'] = getattr(self, 'active_model_name', None)
             
             # Synchronize speed from comms back to state for UI view
             if hasattr(self, 'comms') and self.comms:
                  db = self.comms.robot.deadband_threshold
                  ms = self.comms.robot.max_safe_speed
                  # Show raw PWM in UI
                  self.state['base_speed'] = int(self.comms.base_speed)
                  self.state['turn_speed'] = int(self.comms.turn_speed)
                  
                  # Format: "l:0.00;r:0.00;" or "MOCK: l:0.00;r:0.00;"
             # Motor State Parsing (Assuming format like "l:255;r:255;")
             if motor_cmd_str and getattr(self, 'comms', None) and not self.is_simulation:
                  # Ensure we get clean strings
                  clean_str = motor_cmd_str.replace("MOCK:", "").strip()
                  try:
                      parts = clean_str.split(';')
                      l_val = 0.0
                      r_val = 0.0
                      for p in parts:
                          if 'l:' in p: self.state['motor_l'] = float(p.split(':')[1])
                          if 'r:' in p: self.state['motor_r'] = float(p.split(':')[1])
                  except Exception:
                      self.state['motor_l'] = 0.0
                      self.state['motor_r'] = 0.0
             else:
                  self.state['motor_l'] = 0.0
                  self.state['motor_r'] = 0.0

             if self.is_simulation and hasattr(self, 'sim') and self.sim:
                  telemetry = self.sim.telemetry
                  self.state['sensor_dist'] = telemetry.get('dist', '0')
                  self.state['sensor_batt'] = telemetry.get('batt', '0')
                  self.state['ping'] = telemetry.get('ping', '0')
             elif hasattr(self, 'comms') and self.comms and hasattr(self.comms, 'telemetry'):
                  self.state['sensor_dist'] = self.comms.telemetry.get('dist', '0')
                  self.state['sensor_batt'] = self.comms.telemetry.get('batt', '0')
                  self.state['ping'] = self.comms.telemetry.get('ping', '0')
        
        return new_last_goal_idx

    def _run_loop(self):
        target_dt = 1.0 / CONTROL_FREQ
        steps = 0
        last_goal_idx = -1
        last_sent_time = time.time()
        last_action_sent = (0, 0)
        
        while self.running:
            t_start = time.time()

            try:
                # --- PROCESS COMMAND QUEUE INDEPENDENTLY ---
                with self.queue_lock:
                    while self.command_queue:
                        c_type, c_load = self.command_queue.pop(0)
                        self.dispatcher.dispatch(c_type, c_load)

                img, z_cur, webcam_img = self._sense()
                if img is not None:
                    current_mode = self.state['mode']
                    
                    display_img = img
                    annotated_webcam_img = webcam_img.copy() if webcam_img is not None else None
                    # [NEW] Live Telemetry Proxy Injection
                    feats = None
                    input_dim = 0
                    if hasattr(self, 'planner') and self.planner and getattr(self.planner, 'policy', None):
                        input_layer = getattr(self.planner.policy, 'input_layer', None)
                        input_dim = getattr(input_layer, 'in_features', 0) if input_layer else 0
                        
                    if getattr(self, 'telemetry_extractor', None) and webcam_img is not None and self.use_webcam and getattr(self, 'telemetry_initialized', False):
                        try:
                            gray = cv2.cvtColor(webcam_img, cv2.COLOR_BGR2GRAY)
                            feats = self.telemetry_extractor.process_single_frame(gray)
                            
                            if annotated_webcam_img is not None and feats:
                                cx, cy = int(feats['raw_cx']), int(feats['raw_cy'])
                                import math
                                yaw_rad = math.atan2(feats['sin_yaw'], feats['cos_yaw'])
                                cv2.circle(annotated_webcam_img, (cx, cy), 15, (255, 0, 255), 2, lineType=cv2.LINE_AA)
                                end_x = int(cx + 40 * math.cos(yaw_rad))
                                end_y = int(cy + 40 * math.sin(yaw_rad))
                                cv2.arrowedLine(annotated_webcam_img, (cx, cy), (end_x, end_y), (255, 0, 255), 3, tipLength=0.3)
                                
                                # Plot Goal
                                if hasattr(self, 'planner') and self.planner:
                                    z_g = None
                                    
                                    if self.planner.goals:
                                        g_idx = self.planner.current_goal_idx
                                        if 0 <= g_idx < len(self.planner.goals):
                                            z_g = self.planner.goals[g_idx]['latent']
                                            
                                    if z_g is not None and len(z_g) == 4:
                                         g_cx = int(z_g[0] * 640)
                                         g_cy = int(z_g[1] * 480)
                                         g_yaw_rad = math.atan2(z_g[3], z_g[2])
                                         goal_color = (0, 255, 0) # Green
                                         cv2.circle(annotated_webcam_img, (g_cx, g_cy), 15, goal_color, 2, lineType=cv2.LINE_AA)
                                         g_end_x = int(g_cx + 40 * math.cos(g_yaw_rad))
                                         g_end_y = int(g_cy + 40 * math.sin(g_yaw_rad))
                                         cv2.arrowedLine(annotated_webcam_img, (g_cx, g_cy), (g_end_x, g_end_y), goal_color, 3, tipLength=0.3)
                        except Exception as e:
                            logging.error(f"Failed live telemetry extraction: {e}")
                            
                    # Pure guardrail: Block XYO arrays if webcam is not authorized
                    if not self.use_webcam:
                        feats = None
                        
                    self.live_telemetry_cache = feats

                    current_algo = self.explorer.current_algo if getattr(self, 'explorer', None) else None
                    
                    telemetry_cur = None
                    if feats is not None:
                        telemetry_cur = np.array([
                            feats['cx_norm'],
                            feats['cy_norm'],
                            feats['cos_yaw'],
                            feats['sin_yaw']
                        ], dtype=np.float32)

                    if current_algo == "Algorithmic Oracle":
                        pass # keep z_cur as latent representation
                    elif hasattr(self, 'planner') and self.planner and self.planner.policy:
                        if input_dim in [8, 12, 16, 108, 144] and feats is not None and z_cur is not None:
                            try:
                                t_arr = telemetry_cur
                                
                                if input_dim in [8, 12, 16]:
                                    z_cur = t_arr # Pure Oracle Bypass
                                else:
                                    z_cur = np.concatenate([z_cur.flatten(), t_arr]) # Injection
                            except Exception as e:
                                logging.error(f"Failed live telemetry processing: {e}")
                                if input_dim in [8, 12, 16]: z_cur = np.zeros(4, dtype=np.float32)
                                else: z_cur = np.concatenate([z_cur.flatten(), np.zeros(4, dtype=np.float32)])
                        elif input_dim in [8, 12, 16] and feats is None and z_cur is not None:
                            z_cur = np.zeros(4, dtype=np.float32)
                        elif input_dim in [108, 144] and feats is None and z_cur is not None:
                            z_cur = np.concatenate([z_cur.flatten(), np.zeros(4, dtype=np.float32)])
                    
                    # Decide on action only if z_cur is available
                    target_action, dist, goal_idx, reflex_triggered, is_bout_start = self._decide(current_mode, z_cur, img, last_action_sent, telemetry_cur=telemetry_cur)
                    
                    # --- Model Control Evaluation Logic ---
                    is_model_driving = (self.active_model_name is not None and current_mode in ['LIVE', 'INFERENCE'])
                    is_cql_fixed_goal = (is_model_driving and ('fixed_goal' in self.active_model_name or 'discrete_cql' in self.active_model_name))
                    
                    current_algo = getattr(self.explorer, 'current_algo', None) if getattr(self, 'explorer', None) else None
                    is_algo_oracle = (current_algo == "Algorithmic Oracle")
                    
                    is_fixed_goal = is_cql_fixed_goal or is_algo_oracle
                    
                    if is_fixed_goal:
                        if self.fg_eval_phase == 'MODEL':
                            if is_bout_start:
                                self.fg_eval_bouts += 1
                                
                            if is_algo_oracle:
                                oracle_ctrl = getattr(self.explorer, 'algo_oracle', None)
                                active_pacer = oracle_ctrl.pacer if oracle_ctrl else None
                                active_name = "Algorithmic_Oracle"
                            else:
                                active_pacer = self.cql_controller
                                active_name = self.active_model_name
                                
                            pacer_act = getattr(active_pacer, 'current_action_id', 0)
                            
                            if not self.fg_eval_reached:
                                reached_via_stop = (pacer_act == 5)
                                reached_via_dist = (dist is not None and dist <= self.stop_threshold)
                                
                                if is_algo_oracle:
                                    reached_via_dist = False # Algorithmic Oracle relies on ground truth tracking, not latent distance
                                
                                if reached_via_stop or reached_via_dist:
                                    self.fg_eval_reached = True
                                    self.fg_eval_step_count = self.fg_eval_bouts
                                    logging.info(f"Model reached goal in {self.fg_eval_step_count} steps.")
                            
                            if not self.fg_eval_reached and self.fg_eval_bouts >= 50:
                                # Timeout while driving
                                m_coords = self.state.get('goal_manifold_coords', [])
                                c_coord = m_coords[goal_idx] if m_coords and goal_idx < len(m_coords) else None
                                self.cql_eval_results.append(float('nan'))
                                logging.info("Model abandoned goal after 50 steps (Timeout). Switch to Markov.")
                                if self.recording and self.logger:
                                    self.logger.log_goal_event("ABANDONED", goal_idx, c_coord, 50, active_name)
                                self.fg_eval_phase = 'MARKOV'
                                self.fg_eval_bouts = 0
                                self.fg_eval_reached = False
                                self.fg_eval_step_count = float('nan')
                                
                            elif self.fg_eval_reached and self.fg_eval_bouts >= self.fg_eval_step_count + 10:
                                # Reached goal, and dwelled for 10 steps
                                m_coords = self.state.get('goal_manifold_coords', [])
                                c_coord = m_coords[goal_idx] if m_coords and goal_idx < len(m_coords) else None
                                self.cql_eval_results.append(self.fg_eval_step_count)
                                logging.info(f"Model target finished smoothly in {self.fg_eval_step_count} steps after 10s dwell. Switch to Markov.")
                                if self.recording and self.logger:
                                    self.logger.log_goal_event("REACHED", goal_idx, c_coord, self.fg_eval_step_count, active_name)
                                self.fg_eval_phase = 'MARKOV'
                                self.fg_eval_bouts = 0
                                self.fg_eval_reached = False
                                self.fg_eval_step_count = float('nan')
                                
                        elif self.fg_eval_phase == 'MARKOV':
                            # Override target_action to let Markov randomizer reposition robot
                            try:
                                markov_sonar = float(self.state.get('sensor_dist', 0))
                            except ValueError:
                                markov_sonar = 0.0
                            
                            markov_prev = getattr(self.explorer.markov, 'state', None)
                            target_action = self.explorer.markov.get_action(markov_sonar)
                            markov_new = getattr(self.explorer.markov, 'state', None)
                            
                            is_bout_start = (markov_new == 'MOVE' and markov_prev in ['STOP', 'WAITING'])
                            if is_bout_start:
                                self.fg_eval_bouts += 1
                                
                            # Ensure the robot is sufficiently far from the goal before yielding back
                            min_start_dist = self.stop_threshold * 1.5
                                
                            if self.fg_eval_bouts >= 10 and (dist is None or dist > min_start_dist): # Switch back to MODEL
                                logging.info(f"Markov random positioner finished {self.fg_eval_bouts} steps (Dist: {dist:.2f} > {min_start_dist:.2f}). Switching back to Fixed Goal model.")
                                self.fg_eval_phase = 'MODEL'
                                self.fg_eval_bouts = 0
                                
                                if self.cql_controller:
                                    self.cql_controller.state = 'WAITING'
                                    self.cql_controller.state_start_time = 0
                                    self.cql_controller.current_action_id = 0
                                    if hasattr(self.cql_controller, 'action_queue'):
                                        self.cql_controller.action_queue = []
                                        
                                if is_algo_oracle and getattr(self.explorer, 'algo_oracle', None):
                                    oracle_pacer = getattr(self.explorer.algo_oracle, 'pacer', None)
                                    if oracle_pacer:
                                        oracle_pacer.state = 'WAITING'
                                        oracle_pacer.state_start_time = 0
                                        oracle_pacer.current_action_id = 0
                                        if hasattr(oracle_pacer, 'action_queue'):
                                            oracle_pacer.action_queue = []
                                
                                if self.recording and self.logger:
                                     m_coords = self.state.get('goal_manifold_coords', [])
                                     c_coord = m_coords[goal_idx] if m_coords and goal_idx < len(m_coords) else None
                                     self.logger.log_goal_event("SELECTED", goal_idx, c_coord, 0, self.active_model_name)
                    # ----------------------------
                    # --- Oracle Analytics Logic ---
                    current_algo = getattr(self.explorer, 'current_algo', None)
                    is_oracle = isinstance(current_algo, str) and ("Oracle" in current_algo)

                    if is_oracle and current_algo != "Algorithmic Oracle":
                        oracle_ctrl = None
                        if current_algo.startswith("Neural Oracle"):
                            oracle_ctrl = getattr(self.explorer, 'neural_oracle', None)
                            
                        # If just switched TO Oracle, initialize active state
                        if not self.oracle_analytics['active']:
                            self.oracle_analytics['active'] = True
                            self.oracle_analytics['current_steps'] = 0
                            self.oracle_analytics['consecutive_stops'] = 0
                            self.oracle_analytics['last_goal'] = goal_idx
                            self.oracle_analytics['target_attempts'] = [] # Clean slate for new session
                            
                        if is_bout_start:
                            # Evaluate Goal Swaps
                            if goal_idx != self.oracle_analytics['last_goal']:
                                # New goal selected before previous one was reached
                                if self.oracle_analytics['current_steps'] > 0:
                                    self.oracle_analytics['target_attempts'].append(float('nan'))
                                    
                                self.oracle_analytics['last_goal'] = goal_idx
                                self.oracle_analytics['current_steps'] = 0
                                self.oracle_analytics['consecutive_stops'] = 0

                            # Was action an intentional stop? (act_id == 5)
                            pacer_act = getattr(oracle_ctrl.pacer, 'current_action_id', 0) if oracle_ctrl and hasattr(oracle_ctrl, 'pacer') else 0
                            
                            self.oracle_analytics['current_steps'] += 1
                            
                            if pacer_act == 5:
                                self.oracle_analytics['consecutive_stops'] += 1
                                if self.oracle_analytics['consecutive_stops'] == 5:
                                    # Target Reached! Only log if we weren't ALREADY idling at a target
                                    if not self.oracle_analytics.get('at_goal', False):
                                        self.oracle_analytics['target_attempts'].append(self.oracle_analytics['current_steps'])
                                        self.oracle_analytics['at_goal'] = True
                                        
                                    # Reset internal counts to await next journey
                                    self.oracle_analytics['current_steps'] = 0
                                    self.oracle_analytics['consecutive_stops'] = 0
                            else:
                                self.oracle_analytics['consecutive_stops'] = 0
                                self.oracle_analytics['at_goal'] = False
                    else:
                        self._dump_oracle_analytics()
                    # ----------------------------
                    effective_action, motor_cmd_str, last_action_sent, last_sent_time = self._act(
                        current_mode, target_action, last_action_sent, last_sent_time
                    )
                    
                    ctrl_str = self.active_model_name if is_model_driving else current_algo
                    if is_model_driving and 'fixed_goal' in self.active_model_name:
                         if getattr(self, 'fg_eval_phase', 'MODEL') == 'MARKOV':
                             ctrl_str = 'Markov Random Walk'
                             
                    last_goal_idx = self._update_state(
                        img, z_cur, target_action, effective_action, dist, goal_idx, last_goal_idx, motor_cmd_str,
                        reflex_triggered=reflex_triggered, is_bout_start=is_bout_start, ui_img=display_img, webcam_ui_img=annotated_webcam_img, active_controller_str=ctrl_str
                    )
            
            except (ConnectionResetError, OSError) as e:
                logging.warning(f"Connection Lost ({e}). Reconnecting...")
                self._reconnect_comms()
            except Exception as e:
                import traceback
                logging.error(f"Unexpected Error in Control Loop: {e}")
                traceback.print_exc()
                if "WinError 10054" in str(e) or "ConnectionResetError" in str(e):
                     self._reconnect_comms()
                
            steps += 1
            elapsed = time.time() - t_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            self.state['fps'] = 1.0 / (time.time() - t_start)
