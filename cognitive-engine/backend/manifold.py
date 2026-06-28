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
import pickle
import numpy as np
import cv2
from backend.utils import safe_import_torch
torch = safe_import_torch()

from sklearn.decomposition import PCA
from PIL import Image
import threading
import time

# Use config if available, else defaults
try:
    from config import DATA_DIR, MODELS_DIR
except ImportError:
    DATA_DIR = "./data"
    MODELS_DIR = "./models"

logger = logging.getLogger("ManifoldService")

class ManifoldService:
    def __init__(self, vision_system=None):
        self.vision = vision_system
        self.pca = None
        self.manifold_points = [] # List of [x, y, luminance, action]
        self.library_latents = None # numpy array (N, 64)
        self.library_paths = []     # List of file paths
        self.is_ready = False
        self.lock = threading.Lock()
        self.processing_thread = None
        
        # Determine paths
        self.cache_path = None # Set dynamically based on model name
        self.data_dir = DATA_DIR

    def set_model_name(self, model_filename, model_path=None):
        """Sets the cache path based on the loaded model filename."""
        self.active_model_path = model_path # [NEW] Store full path for stale check

        # [FIX] Scope data directory to the model's dataset
        if model_path:
            parent_dir = os.path.dirname(model_path)
            # Check if we are in a subdirectory of data (heuristic)
            # If the model is in 'data/nook', we want 'data/nook'
            # If the model is in 'models/', we default to 'data/' (global)
            
            # Simple check: Is valid dir and NOT the global models dir?
            if os.path.isdir(parent_dir) and os.path.normpath(parent_dir) != os.path.normpath(MODELS_DIR):
                 self.data_dir = parent_dir
                 logger.info(f"Manifold Data Scope restricted to: {self.data_dir}")
            else:
                 self.data_dir = DATA_DIR
                 logger.info(f"Manifold Data Scope set to Global: {self.data_dir}")
        else:
            self.data_dir = DATA_DIR

        if not model_filename:
            self.cache_path = os.path.join(MODELS_DIR, "manifold.pkl")
            return

        base = os.path.basename(model_filename)
        name, ext = os.path.splitext(base)
        cache_filename = f"{name}_manifold.pkl"

        # 1. Default: models/[name]_manifold.pkl
        default_cache = os.path.join(MODELS_DIR, cache_filename)
        
        # 2. Check for the model-specific cache file in the same directory as the model 
        # (This is where training typically saves it: data/Dataset/[name]_manifold.pkl)
        sibling_cache = None
        if model_path:
            parent_dir = os.path.dirname(model_path)
            sibling_check = os.path.join(parent_dir, cache_filename)
            if os.path.exists(sibling_check):
                sibling_cache = sibling_check
        
        # Priority: Sibling (Training artifact) > Default (Runtime cache)
        if sibling_cache:
             self.cache_path = sibling_cache
             logger.info(f"Manifold Cache set to (Sibling): {self.cache_path}")
        else:
             # Force creation in sibling dir if model_path is provided
             if model_path:
                 parent_dir = os.path.dirname(model_path)
                 # Heuristic: Is this a data dir?
                 if "data" in parent_dir or "Dataset" in parent_dir or os.path.basename(os.path.dirname(parent_dir)) == "data":
                     self.cache_path = os.path.join(parent_dir, cache_filename)
                     logger.info(f"Manifold Cache targeting (Sibling): {self.cache_path}")
                 else:
                     self.cache_path = default_cache
                     logger.info(f"Manifold Cache set to (Default): {self.cache_path}")
             else:
                 self.cache_path = default_cache
                 logger.info(f"Manifold Cache set to (Default): {self.cache_path}")

    def init_from_cache(self):
        """Attempts to load from cache. Does NOT trigger generation if missing."""
        if self._load_from_cache():
             logger.info("Manifold Service initialized from cache.")
        else:
             logger.info("Manifold cache missing or stale. Service disabled until generated.")
             self.is_ready = False

    def start_background_fit(self, force=False):
        """Starts the PCA fitting process in a background thread."""
        if self.processing_thread and self.processing_thread.is_alive():
             logger.warning("Manifold generation already in progress. Skipping request.")
             return

        t = threading.Thread(target=self._fit_process, args=(force,), daemon=True)
        self.processing_thread = t
        t.start()
        
    def fit(self, force=False):
        """Synchronous fit (blocking)."""
        self._fit_process(force=force)

    def _fit_process(self, force=False):
        logger.info(f"Starting Manifold PCA process... (Force={force})")
        
        # 1. Check Cache
        if not force and self._load_from_cache():
            return

        # 2. Collect Data
        if not self.vision:
            logger.warning("No VisionSystem provided. Cannot compute manifold.")
            return

        latents, paths, luminances, actions = self._collect_latents()
        
        if len(latents) < 10:
            logger.debug(f"Not enough data points ({len(latents)}) for robust PCA. Need at least 10.")
            self.is_ready = False
            return

        # 3. Fit PCA
        try:
            logger.info(f"Fitting PCA on {len(latents)} points...")
            
            # [DEBUG] Check latency range
            l_min, l_max = np.min(latents), np.max(latents)
            l_mean, l_std = np.mean(latents), np.std(latents)
            logger.info(f"[DEBUG] Latents Stats | Min: {l_min:.4f} | Max: {l_max:.4f} | Mean: {l_mean:.4f} | Std: {l_std:.4f}")

            self.pca = PCA(n_components=2)
            transformed = self.pca.fit_transform(latents)
            
            # Save results
            with self.lock:
                pts = []
                for i in range(len(transformed)):
                    pts.append([
                        float(transformed[i, 0]), 
                        float(transformed[i, 1]),
                        luminances[i],
                        actions[i]
                    ])
                self.manifold_points = pts
                self.library_latents = latents
                self.library_paths = paths
                
                # [NEW] Calculate Bounds
                xs = transformed[:, 0]
                ys = transformed[:, 1]
                # Add 10% padding
                pad_x = (np.max(xs) - np.min(xs)) * 0.1
                pad_y = (np.max(ys) - np.min(ys)) * 0.1
                self.bounds = [float(np.min(xs) - pad_x), float(np.max(xs) + pad_x), 
                               float(np.min(ys) - pad_y), float(np.max(ys) + pad_y)]
                               
                self.is_ready = True
                
                # [NEW] Generate and Save Background Image
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # Create figure with white background
                fig = plt.figure(figsize=(8, 8), dpi=100)
                fig.patch.set_facecolor('white')
                
                ax = plt.gca()
                ax.set_facecolor("white")
                
                # Scatter with slate color for "cloud" look
                # Color: #64748b (Slate 500) matches UI
                # Dots larger as requested (s=100) -> Reduced 25% = 75
                plt.scatter(transformed[:, 0], transformed[:, 1], c='#64748b', s=75, alpha=0.3, edgecolors='none')
                
                # Use calculated bounds
                plt.xlim(self.bounds[0], self.bounds[1])
                plt.ylim(self.bounds[2], self.bounds[3])
                
                plt.axis('off') # No axes, just the cloud
                plt.tight_layout(pad=0)
                
                # Ensure cache_path is string for replacement
                cache_path_str = str(self.cache_path)
                img_path = cache_path_str.replace('.pkl', '.png')
                plt.savefig(img_path, format='png', transparent=False, bbox_inches='tight', pad_inches=0, facecolor='white')
                plt.close(fig) # Explicit close with fig
                logger.info(f"Saved manifold background image to {img_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate manifold image: {e}")

            self._save_to_cache()
            logger.info("Manifold PCA ready.")
            
        except Exception as e:
            logger.error(f"PCA Fit failed: {e}")

    def _collect_latents(self):
        """Scans the pre-extracted goals directory for images and runs them through the encoder."""
        latents = []
        paths = []
        
        if not self.data_dir: 
            return [], []
            
        import glob
        import shutil
        
        # Extract base VAE name, safely stripping any '-cql_' tag if present
        model_filename = getattr(self, 'active_model_path', getattr(self, 'model_filename', None))
        model_name, _ = os.path.splitext(os.path.basename(model_filename)) if model_filename else ("default", "")
        if '-cql_' in model_name:
            model_name = model_name.split('-cql_')[0]
        
        if model_name in [None, "default", "N/A", ""]:
             logger.debug(f"Skipping manifold generation; no valid model loaded (name: {model_name})")
             return [], [], [], []

        sources = []
        transitions_path = os.path.join(self.data_dir, "all_transitions.json")
        import json
        import random
        
        if os.path.exists(transitions_path):
            try:
                with open(transitions_path, 'r') as f:
                    transitions = json.load(f)
                logger.info(f"Manifold Gen: Found {len(transitions)} transitions in {transitions_path}")
                # Sample up to 1000 robustly
                samples = random.sample(transitions, min(1000, len(transitions)))
                for t in samples:
                    sources.append({
                        'format': 'transition',
                        'path': os.path.join(self.data_dir, t['image_path']),
                        'action': t.get('macro_action', 0)
                    })
            except Exception as e:
                logger.error(f"Failed to load transitions: {e}")

        if not sources:
            model_goals_dir = os.path.join(self.data_dir, f"{model_name}_goals")
            goals_dir = os.path.join(self.data_dir, "goals")
            
            if os.path.exists(model_goals_dir) and len(glob.glob(os.path.join(model_goals_dir, "*.jpg"))) > 0:
                goals_dir = model_goals_dir
                logger.info(f"Using isolated goal cache: {goals_dir}")
            elif os.path.exists(goals_dir):
                try:
                    shutil.move(goals_dir, model_goals_dir)
                    goals_dir = model_goals_dir
                    logger.info(f"Isolated goal library for {model_name} into {model_goals_dir}")
                except Exception as e:
                    logger.error(f"Failed to isolate goals to {model_goals_dir}: {e}")
            else:
                logger.warning(f"Goals directory not found at {goals_dir} or {model_goals_dir}. Cannot generate manifold for {model_name}.")
                return [], [], [], []
                
            # Collect static JPGs
            files = glob.glob(os.path.join(goals_dir, "*.jpg"))
            files.sort()
            
            sources = [{'format': 'standard', 'path': f, 'action': 0} for f in files]
            logger.info(f"Manifold Gen: Found {len(sources)} cached goal images in {goals_dir}")

        if len(sources) < 100:
            logger.warning(f"Very few images to generate manifold ({len(sources)}). Proceeding anyway, but quality may be poor.")
        
        # Use explicitly up to 1000
        selected_sources = sources[:1000]

        count = 0
        self.vision.encoder.eval()
        
        batch_size = 64
        
        luminances = []
        actions = []
        
        if torch:
            with torch.no_grad():
                # Process in batches
                for i in range(0, len(selected_sources), batch_size):
                    batch_sources = selected_sources[i:i + batch_size]
                    batch_tensors = []
                    valid_paths = []
                    valid_sources = []
                    
                    for source in batch_sources:
                        path_to_store = source['path']
                        try:
                            # UI projection explicitly locked to pure egocentric vision
                                
                            if hasattr(self.vision, 'process_frame'):
                                img = cv2.imread(path_to_store)
                                if img is None: continue
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_pil = Image.fromarray(img)
                            else:
                                img_pil = Image.open(path_to_store).convert('RGB')
                                
                            if self.vision.transform is None: 
                                continue
                                
                            try:
                                input_tensor = self.vision.transform(img_pil)
                            except TypeError:
                                input_tensor = self.vision.transform(np.array(img_pil))
                                    
                            batch_tensors.append(input_tensor)
                            valid_paths.append(path_to_store)
                            valid_sources.append({
                                'source': source,
                                'img_pil': img_pil
                            })
                            
                        except Exception as e:
                            logger.error(f"Failed to prepare {source['path']}: {e}")
                            
                    if not batch_tensors:
                        continue
                        
                    try:
                        batch_tensor = torch.stack(batch_tensors).to(self.vision.device)
                        
                        ret = self.vision.encoder(batch_tensor)
                        
                        if ret is None: continue
                        
                        # Handle different return types
                        if isinstance(ret, tuple) or isinstance(ret, list):
                            if len(ret) == 5:
                                _, mu, _, _, _ = ret
                            elif len(ret) == 4:
                                _, mu, _, _ = ret
                            elif len(ret) == 3:
                                _, mu, _ = ret
                            elif len(ret) == 2:
                                _, mu = ret
                            else:
                                mu = ret[0] if len(ret) > 0 else None
                        else:
                            mu = ret
                            
                        if mu is None: continue
                        
                        # Handle DummyVisionSLAM MockEncoder (which returns (None, mu, None))
                        if isinstance(mu, tuple):
                             mu = mu[1]
                             
                        mu_np = mu.cpu().numpy()
                        
                        for idx, valid_path in enumerate(valid_paths):
                            latents.append(mu_np[idx].flatten())
                            paths.append(valid_path)
                            
                            # Calculate specific properties natively
                            try:
                                img_np = np.array(valid_sources[idx]['img_pil']).astype(np.float32) / 255.0
                                h = img_np.shape[0]
                                top_half = img_np[:h//2, :, :]
                                lum = 0.299 * top_half[:, :, 0] + 0.587 * top_half[:, :, 1] + 0.114 * top_half[:, :, 2]
                                luminances.append(float(lum.mean()))
                            except Exception as em:
                                luminances.append(0.5)
                            
                            actions.append(valid_sources[idx]['source'].get('action', 0))
                            
                        count += len(valid_paths)
                        if count % 100 < len(valid_paths): # Log roughly every 100
                            logger.debug(f"Encoded {count}/{len(selected_sources)}")
                            
                    except Exception as e:
                        logger.error(f"Batch encoding failed: {e}")
                    
        return np.array(latents), paths, luminances, actions

    def _save_to_cache(self):
        if not self.cache_path:
             logger.warning("Cannot save cache: cache_path not set.")
             return
             
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'pca': self.pca,
                    'latents': self.library_latents,
                    'points': self.manifold_points,
                    'paths': self.library_paths,
                    'bounds': getattr(self, 'bounds', None)
                }, f)
            logger.info(f"Saved manifold cache to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_from_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            logger.info(f"Cache miss: Path {self.cache_path} does not exist.")
            return False
            
        try:
            # [NEW] Check timestamp against model
            model_path = getattr(self, 'active_model_path', None)

            # Fallback heuristic if no path explicitly set
            if not model_path:
                for m in ["tiny_vae_final.pth", "vae_encoder.pth"]:
                    p = os.path.join(MODELS_DIR, m)
                    if os.path.exists(p):
                        model_path = p
                        break
            
            if model_path and os.path.exists(model_path):
                cache_mtime = os.path.getmtime(self.cache_path)
                model_mtime = os.path.getmtime(model_path)
                if model_mtime > cache_mtime:
                    logger.warning(f"Manifold cache stale ({self.cache_path}). Model {os.path.basename(model_path)} newer.")
                    logger.warning(f"Model time: {model_mtime}, Cache time: {cache_mtime}, Diff: {model_mtime - cache_mtime}")
                    return False

            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                
            if 'latents' not in data or 'paths' not in data or 'bounds' not in data:
                logger.warning("Cache is missing 'latents', 'paths', or 'bounds'. Ignoring cache.")
                return False

            with self.lock:
                self.pca = data['pca']
                self.manifold_points = data.get('points', [])
                self.library_latents = data.get('latents', None)
                self.library_paths = data.get('paths', [])
                self.bounds = data.get('bounds', None) # [NEW] Load Bounds
                self.is_ready = True
            
            logger.info(f"Loaded Manifold from cache: {self.cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def project(self, z):
        """Projects a single latent vector (64,) to 2D."""
        if not self.is_ready or self.pca is None:
            return None
        
        try:
            # PCA expects (n_samples, n_features)
            z_in = z.reshape(1, -1)
            
            # [NEW] Check if z dimension matches PCA expectation
            if self.pca and z_in.shape[1] != self.pca.n_features_in_:
                logger.warning(f"Manifold Project: Dimension mismatch. Latent {z_in.shape[1]} != PCA {self.pca.n_features_in_}. Fit required.")
                return None
                
            res = self.pca.transform(z_in)
            
            if np.isnan(res).any():
                logger.warning("Manifold Project: NaN detected in result")
                return None
                
            return res[0].tolist() # [x, y]
        except Exception as e:
            logger.warning(f"Manifold Project Error: {e}")
            return None

    # def get_nearest_image(self, z):
    #     """
    #     Finds the nearest neighbor in the latent library and returns its B64 image string.
    #     Disabled per user request.
    #     """
    #     return None
