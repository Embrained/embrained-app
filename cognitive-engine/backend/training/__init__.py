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
import csv
import re
import random
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
import importlib.util
import io
import base64
import matplotlib
matplotlib.use('Agg') # [Fix] Ensure backend is thread-safe
import matplotlib.pyplot as plt

# Logger setup
logger = logging.getLogger("TrainingPipeline")

# Constants
import sys

MIN_ACTION_STEPS = 2
MAX_GOAL_DISTANCE_HORIZON = 50
MAX_GOALS_PER_START_FRAME = 3
STOP_ACTION = (0, 0)
MOTOR_RE = re.compile(r'l: *(-?\d+);r: *(-?\d+);')

# [Fix] Define MODELS_DIR
MODELS_DIR = Path(os.getcwd()) / "models"

class TrainingPipeline:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.lock = threading.RLock()
        # [NEW] Plot Update Tracking
        self.last_plot_time = 0
        self.plot_interval = 20 # seconds
        self.is_generating_plot = False
        
    def _expand_datasets(self, selected_prefixes):
        """
        Takes a list of dataset prefixes (as selected in the UI) and returns a list of
        all actual dataset directory names in data_root that match those prefixes.
        """
        if not selected_prefixes:
            return []
            
        import re
        expanded_names = set()
        
        # We need to list all directories in data_root
        if not self.data_root.exists():
            return []
            
        all_dirs = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        
        for prefix in selected_prefixes:
            # First, check for exact match in case it's not a prefix
            if prefix in all_dirs:
                expanded_names.add(prefix)
                
            # Then check for prefix pattern match
            pattern = re.compile(rf'^{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})$')
            for d in all_dirs:
                if pattern.match(d):
                    expanded_names.add(d)
                    
        return sorted(list(expanded_names))
        
    def process_datasets(self, dataset_names, extract_goals=True, extract_telemetry=True):
        """
        Main entry point to process selected datasets.
        """
        if not self.lock.acquire(blocking=False):
            return {"status": "error", "message": "Already processing"}

        try:
            expanded_datasets = self._expand_datasets(dataset_names)
            if not expanded_datasets:
                return {"status": "error", "message": "No matching dataset directories found."}
                
            # Step 1: Parse Transitions
            all_transitions = self._parse_sessions(expanded_datasets, extract_goals=extract_goals)
            
            # Save transitions (intermediate)
            transitions_path = self.data_root / "all_transitions.json"
            with open(transitions_path, 'w') as f:
                json.dump(all_transitions, f, indent=2)
                
            # Step 2: Auto-Extract Allocentric Global Telemetry Matrix (if explicitly requested)
            if extract_telemetry:
                logger.info("Executing Verbose Offline Telemetry Analysis across parsed datasets...")
                if expanded_datasets:
                    try:
                        from scripts.extract_telemetry import TelemetryExtractor
                        # Convert to absolute paths for the extractor
                        abs_dirs = [(self.data_root / d).as_posix() for d in expanded_datasets]
                        extractor = TelemetryExtractor(abs_dirs)
                        
                        # Generate dataframe
                        df_telemetry = extractor.process_all()
                        
                        telemetry_path = self.data_root / "master_telemetry.csv"
                        df_telemetry.to_csv(telemetry_path.as_posix(), index=False)
                        logger.info(f"Successfully compiled {len(df_telemetry)} geometric coordinates to {telemetry_path}")
                        
                        # [NEW] Automatically evaluate metrics and generate analytical plots
                        try:
                            from scripts.visualize_telemetry import generate_telemetry_plots
                            import time
                            
                            ts_prefix = time.strftime("%Y%m%d_%H%M%S")
                            
                            # Grab prefix (e.g. markov) safely
                            if dataset_names and len(dataset_names) == 1:
                                target_id = dataset_names[0]
                            else:
                                ds_bases = [d for d in expanded_datasets]
                                if len(ds_bases) == 1:
                                    target_id = os.path.basename(ds_bases[0])
                                else:
                                    target_id = dataset_names[0] if dataset_names else "mixed"
                                
                            plot_prefix = f"{ts_prefix}_{target_id}"
                            generate_telemetry_plots(self.data_root.as_posix(), plot_prefix)
                        except Exception as plt_err:
                            logger.warning(f"Failed to generate automated telemetry analytics: {plt_err}")
                            
                    except Exception as eval_err:
                        logger.debug(f"Telemetry extraction skipped or failed (Likely no webcam overhead frames available): {eval_err}")
                    
                return {
                    "status": "success", 
                    "transitions_count": len(all_transitions),
                    "message": f"Processed {len(all_transitions)} transitions and compiled telemetry mappings."
                }
            else:
                return {
                    "status": "success", 
                    "transitions_count": len(all_transitions),
                    "message": f"Processed {len(all_transitions)} transitions."
                }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.lock.release()

    
    def run_vae_pipeline(self, num_epochs=20, stop_event=None, progress_callback=None, batch_size=64, learning_rate=1e-4, beta=0.5, model_size='large', selected_datasets=None, model_filename=None, architecture='continuous', latent_dim=128):      
        """
        Executes VAE Training + Manifold Regeneration.
        """
        if not self.lock.acquire(blocking=False):
            return {"status": "error", "message": "Already processing"}

        try:
            # Expand datasets first
            expanded_datasets = self._expand_datasets(selected_datasets) if selected_datasets else None
            
            # Step 0: Ensure we have transitions if specific datasets selected
            if expanded_datasets:
                logger.debug(f"Preparing temporary transitions for VAE on expanded datasets: {expanded_datasets}")
                self.process_datasets(selected_datasets, extract_goals=False, extract_telemetry=False) # Pass unexpanded since process_datasets auto-expands

            # Step 1: Train VAE
            logger.info("Starting VAE Training...")
            
            # [NEW] Create a Plot Helper with Throttling
            last_plot_time = [0.0] # Use list for closure mutability [FIX] Force initial plot
            last_msg = {'loss': 0.0, 'kld': 0.0}
            last_epoch_seen = [-1] # [FIX] Track integer epochs
            history = []
            
            def plot_aware_callback(epoch, loss, kld=0.0, checkpoint_ready=None, manifold_plot=None, recon=0.0, **kwargs):
                last_msg['loss'] = loss
                last_msg['kld'] = kld
                history.append({"epoch": epoch, "loss": loss, "kld": kld, "kl": kld, "recon": recon})
                
                # Signal engine state
                now = time.time()
                current_ep = int(epoch)
                
                # [FIX] Report every 5s or at integer epoch boundaries to avoid float modulo issues
                if (now - last_plot_time[0]) >= 5.0 or current_ep > last_epoch_seen[0] or manifold_plot:
                    last_plot_time[0] = now
                    last_epoch_seen[0] = current_ep
                    if progress_callback:
                        logger.debug(f"Reporting Progress: Epoch {epoch:.2f}, Loss {loss:.4f}")
                        try:
                            progress_callback(epoch, loss, kld=kld, recon=recon, manifold_plot=manifold_plot, **kwargs)
                        except Exception as e:
                            logger.error(f"Error in progress_callback: {e}")
                
                return {}

            from backend.train_vae import train as train_vae_model
            model_path_str = train_vae_model(
                self.data_root, 
                num_epochs=num_epochs, 
                stop_event=stop_event, 
                progress_callback=plot_aware_callback,
                batch_size=batch_size,
                learning_rate=learning_rate,
                beta=beta,
                model_size=model_size,
                selected_datasets=expanded_datasets,
                model_filename=model_filename,
                architecture=architecture,
                latent_dim=latent_dim
            )

            # Step 2: Regenerate Manifold
            # (Required for visualizations to match new latent space)
            # Determine model path
            if model_path_str:
                model_path = Path(model_path_str)
            else:
                # Fallback: Search for the latest VAE model in data_root or models dir
                import glob
                candidates = glob.glob(str(self.data_root / "*-vae_*.pth")) + glob.glob(str(MODELS_DIR / "*-vae_*.pth"))
                if not candidates:
                    candidates = [str(self.data_root / "tiny_vae_final.pth"), str(MODELS_DIR / "tiny_vae_final.pth")]
                
                # Sort by modification time to get the latest
                candidates = [c for c in candidates if os.path.exists(c)]
                if candidates:
                    candidates.sort(key=os.path.getmtime, reverse=True)
                    model_path = Path(candidates[0])
                else:
                    model_path = self.data_root / "tiny_vae_final.pth" # Final fallback
                 
            self._regenerate_manifold(model_path) # Uses the new model immediately
            
            # [NEW] Generate and Save Manifold Plot for UI
            try:
                # We need to add root to sys path to import verify_manifold if needed, 
                # but typically backend modules are importable.
                import sys
                root_dir = str(self.data_root.parent)
                if root_dir not in sys.path:
                     sys.path.append(root_dir)
                
                # Dynamic import to avoid top-level issues
                from analysis_archive.verify_manifold import run_sanity_check
                img_b64 = run_sanity_check(data_path=str(self.data_root), model_path=str(model_path), force_recompute=True, selected_datasets=expanded_datasets)
                if img_b64 and progress_callback:
                    logger.debug("Sending final manifold plot to UI.")
                    progress_callback(num_epochs, last_msg['loss'], last_msg['kld'], manifold_plot=img_b64)
                logger.debug("Manifold plot cache updated.")
            except Exception as e:
                logger.error(f"Failed to update manifold plot cache: {e}")
                
            try:
                meta = {
                    "type": "vae",
                    "hyperparameters": {
                        "epochs": num_epochs,
                        "batchSize": batch_size,
                        "learningRate": learning_rate,
                        "beta": beta,
                        "modelSize": model_size,
                        "pipelineArchitecture": architecture,
                        "latentDim": latent_dim
                    },
                    "history": history
                }
                meta_path = str(model_path).replace(".pth", "_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f)
            except Exception as e:
                logger.error(f"Failed to save model metadata: {e}")

            return {"status": "success", "message": "VAE Trained and Manifold Updated", "manifold_plot": img_b64 if 'img_b64' in locals() else None}
            
        except Exception as e:
            logger.error(f"VAE Pipeline failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.lock.release()

    def _regenerate_manifold(self, model_path, model_type="vae"):
        """Helper to regenerate manifold cache."""
        logger.debug(f"Regenerating Manifold using {model_path} ({model_type})...")
        try:
            # We need to reload/import to ensure freshness
            from backend.manifold import ManifoldService
            
            import torch
            
            device = 'cuda' if importlib.util.find_spec("torch") and torch.cuda.is_available() else 'cpu'
            
            vision_system = None
            if model_type == "vae":
                from modules.vision import VisionSystem
                vision_system = VisionSystem(device=device, model_path=str(model_path))
            elif model_type == "latentslam":
                from backend.services.inference_service import LatentSLAMInference
                slam_inf = LatentSLAMInference(str(model_path), device=device)
                
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
                        self.encoder = MockEncoder(slam_inf.model)
                vision_system = DummyVisionSLAM(slam_inf)
            
            manifold = ManifoldService(vision_system)
            manifold.data_dir = self.data_root 
            manifold.set_model_name(os.path.basename(str(model_path)), model_path=str(model_path))
            # [Fix] Save cache to dataset directory (Sibling Cache)
            base_model = os.path.basename(str(model_path))
            model_name, _ = os.path.splitext(base_model)
            manifold.cache_path = self.data_root / f"{model_name}_manifold.pkl"
            
            manifold.fit(force=True) 
            
            logger.debug("Manifold regeneration complete.")
        except Exception as e:
            logger.error(f"Manifold regeneration failed: {e}")

    def run_latentslam_pipeline(self, num_epochs=20, stop_event=None, progress_callback=None, batch_size=64, learning_rate=1e-4, beta=2.0, transition_loss_weight=1.0, contrastive_weight=0.0, architecture="continuous", model_size="large", selected_datasets=None, model_filename=None, image_size=64, num_layers=4, latent_dim=128):
        """
        Executes LatentSLAM (GSSM) Training.
        """
        if not self.lock.acquire(blocking=False):
            return {"status": "error", "message": "Already processing"}

        try:
            logger.info("Starting LatentSLAM Training...")
            
            # Construct full paths for selected datasets
            dataset_dirs = None
            if selected_datasets:
                expanded_datasets = self._expand_datasets(selected_datasets)
                dataset_dirs = [str(self.data_root / ds) for ds in expanded_datasets]

            from backend.training.train_latentslam import train as train_ls_model
            
            # [NEW] Create a Plot Helper with Throttling
            last_plot_time = [0.0] 
            last_msg = {'loss': 0.0, 'kl': 0.0, 'recon': 0.0}
            last_epoch_seen = [-1] 
            history = []
            
            def plot_aware_callback(epoch, loss, kl=0.0, recon=0.0, manifold_plot=None):
                last_msg['loss'] = loss
                last_msg['kl'] = kl
                last_msg['recon'] = recon
                history.append({"epoch": epoch, "loss": loss, "kld": kl, "kl": kl, "recon": recon})
                now = time.time()
                current_ep = int(epoch)
                
                if (now - last_plot_time[0]) >= 5.0 or current_ep > last_epoch_seen[0] or manifold_plot:
                    last_plot_time[0] = now
                    last_epoch_seen[0] = current_ep
                    if progress_callback:
                        progress_callback(epoch, loss, kl=kl, recon=recon, manifold_plot=manifold_plot)

            model_path_str = train_ls_model(
                self.data_root, 
                num_epochs=num_epochs, 
                stop_event=stop_event, 
                progress_callback=plot_aware_callback,
                batch_size=batch_size,
                learning_rate=learning_rate,
                beta=beta,
                transition_loss_weight=transition_loss_weight,
                contrastive_weight=contrastive_weight,
                architecture=architecture,
                model_size=model_size,
                dataset_dirs=dataset_dirs,
                model_filename=model_filename,
                image_size=image_size,
                num_layers=num_layers,
                latent_dim=latent_dim
            )

            # [NEW] Regenerate Manifold and Plot
            if model_path_str:
                model_path = Path(model_path_str)
                self._regenerate_manifold(model_path, model_type="latentslam")

                try:
                    import base64
                    base_model = os.path.basename(str(model_path))
                    model_name, _ = os.path.splitext(base_model)
                    img_path = self.data_root / f"{model_name}_manifold.png"
                    
                    if img_path.exists():
                        with open(img_path, "rb") as f:
                            img_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                        if progress_callback:
                            plot_aware_callback(num_epochs, last_msg['loss'], last_msg['kl'], last_msg['recon'], manifold_plot=img_b64)
                    else:
                        logger.warning(f"No LatentSLAM manifold image found at {img_path}")
                except Exception as e:
                    logger.error(f"Failed to update manifold plot cache: {e}")
                    
                # [NEW] Generate Confusion/Parity Matrix Automatically for UI consumption
                parity_b64 = None
                try:
                    logger.info("Executing Automatic Parity Extrapolation...")
                    from backend.training.evaluate_oracles import run_evaluation
                    run_evaluation() # Outputs strictly to data_root / {name}_parity.png
                    
                    parity_path = self.data_root / f"{model_name}_parity.png"
                    if parity_path.exists():
                        with open(parity_path, "rb") as f:
                            parity_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                except Exception as e:
                    logger.error(f"Failed to auto-generate parity plot: {e}")
                    
                try:
                    meta = {
                        "type": "latentslam",
                        "hyperparameters": {
                            "epochs": num_epochs,
                            "batchSize": batch_size,
                            "learningRate": learning_rate,
                            "modelSize": model_size,
                            "imageSize": image_size,
                            "numLayers": num_layers,
                            "latentDim": latent_dim,
                            "forwardApproach": "latentslam",
                            "transitionLossWeight": transition_loss_weight,
                            "contrastiveWeight": contrastive_weight,
                            "pipelineArchitecture": architecture,
                            "beta": beta
                        },
                        "history": history
                    }
                    meta_path = str(model_path).replace(".pth", "_meta.json")
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=4)
                        
                    result = {"status": "success", "message": "LatentSLAM specific pipeline finished", "model": str(model_path), "metadata": meta}
                    if parity_b64:
                        result["policy_heatmap"] = parity_b64
                    return result
                except Exception as e:
                    logger.error(f"Failed to save metadata: {e}")

            return {"status": "success", "message": "LatentSLAM Trained", "model_path": model_path_str, "manifold_plot": img_b64 if 'img_b64' in locals() else None}
            
        except Exception as e:
            logger.error(f"LatentSLAM Pipeline failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.lock.release()

    def run_dreamer_pipeline(self, num_epochs=50, stop_event=None, progress_callback=None, batch_size=32, learning_rate=1e-4, tag="red_ball", selected_datasets=None, model_filename=None):
        """
        Executes DreamerV3 World Model Training.
        """
        if not self.lock.acquire(blocking=False):
            return {"status": "error", "message": "Already processing"}

        try:
            logger.info("Starting DreamerV3 World Model Training...")
            
            # Construct full paths for selected datasets
            dataset_dirs = None
            if selected_datasets:
                expanded_datasets = self._expand_datasets(selected_datasets)
                dataset_dirs = [str(self.data_root / ds) for ds in expanded_datasets]

            from backend.training.train_dreamer import train as train_dreamer_model
            model_path_str = train_dreamer_model(
                self.data_root, 
                num_epochs=num_epochs, 
                stop_event=stop_event, 
                progress_callback=progress_callback,
                batch_size=batch_size,
                learning_rate=learning_rate,
                tag=tag,
                dataset_dirs=dataset_dirs,
                model_filename=model_filename
            )

            return {"status": "success", "message": "DreamerV3 Policy Trained", "model_path": model_path_str}
            
        except Exception as e:
            logger.error(f"DreamerV3 Pipeline failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.lock.release()

    def run_topological_pipeline(self, num_epochs=20, stop_event=None, progress_callback=None, batch_size=128, learning_rate=1e-4, approach='mse', selected_datasets=None, model_filename=None):
        """
        Executes Forward Model optimization via MSE or InfoNCE.
        """
        if not self.lock.acquire(blocking=False):
            return {"status": "error", "message": "Already processing"}

        try:
            expanded_datasets = self._expand_datasets(selected_datasets) if selected_datasets else None
            
            if expanded_datasets:
                logger.debug(f"Preparing temporary transitions for Forward Model on expanded datasets: {expanded_datasets}")
                self.process_datasets(selected_datasets, extract_goals=False, extract_telemetry=False) 

            logger.info(f"Starting Forward Model Training ({approach.upper()})...")
            history = []
            
            def intercept_progress(epoch, loss, **kwargs):
                history.append({"epoch": epoch, "loss": loss})
                if progress_callback:
                    progress_callback(epoch, loss, **kwargs)
            
            # Dynamically select training script
            if approach == 'mse':
                from backend.training.train_topological_mse import train as train_forward_model
            elif approach == 'weighted':
                from backend.training.train_topological_weighted import train as train_forward_model
            elif approach == 'rnn':
                from backend.training.train_topological_rnn import train as train_forward_model
            else:
                from backend.training.train_topological_infonce import train as train_forward_model
                
            model_path, parity_plot_path = train_forward_model(
                data_root=self.data_root, 
                num_epochs=num_epochs, 
                stop_event=stop_event, 
                progress_callback=intercept_progress,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_filename=model_filename
            )

            # Send Output Graph back to UI
            if parity_plot_path and os.path.exists(parity_plot_path) and progress_callback:
                try:
                    # Dynamically hook into the Dashboard synthesizer script!
                    import glob
                    from analysis_archive.verify_forward_model import generate_forward_dashboard
                    
                    # Locate the auto-assigned dependent VAE automatically
                    vae_paths = glob.glob(os.path.join(self.data_root, "tinyvae-*.pth"))
                    vae_model_path = sorted(vae_paths, key=os.path.getmtime)[-1] if vae_paths else ""
                    
                    base_model = os.path.basename(model_path) if model_path else 'FallbackModel'
                    
                    encoded = generate_forward_dashboard(
                        fwd_model_path=model_path,
                        vae_model_path=vae_model_path,
                        parity_plot_path=parity_plot_path,
                        data_root=self.data_root,
                        model_name=base_model,
                        approach=approach
                    )
                    
                    # Fallback if synthesis failed 
                    if not encoded or encoded == "":
                        import base64
                        with open(parity_plot_path, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode('utf-8')
                            
                    progress_callback(num_epochs, 0, policy_heatmap=f"data:image/png;base64,{encoded}")
                except Exception as e:
                    logger.error(f"Failed to synthesize and evaluate diagnostic dashboard parity layout: {e}")
                    
            try:
                meta = {
                    "type": "forward",
                    "hyperparameters": {
                        "epochs": num_epochs,
                        "batchSize": batch_size,
                        "learningRate": learning_rate,
                        "forwardApproach": approach
                    },
                    "history": history
                }
                meta_path = str(model_path).replace(".pth", "_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f)
            except Exception as e:
                logger.error(f"Failed to save metadata file: {e}")
            
            return {
                "status": "success",
                "model_path": model_path,
                "policy_heatmap": f"data:image/png;base64,{encoded}" if 'encoded' in locals() and encoded else None
            }
            
        except Exception as e:
            logger.error(f"Forward Model Pipeline failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.lock.release()

    def get_policy_confusion_matrix(self, model_filename=None):
        """
        Reads the generated confusion matrix PNG and returns it as a base64 encoded image.
        """
        import base64
        try:
            if model_filename:
                if os.path.exists(model_filename):
                    policy_path = Path(model_filename)
                elif (self.data_root / model_filename).exists():
                    policy_path = self.data_root / model_filename
                elif (MODELS_DIR / model_filename).exists():
                     policy_path = MODELS_DIR / model_filename
                else:
                    return {"status": "error", "message": f"Model {model_filename} not found"}
            else:
                import glob
                candidates = glob.glob(str(self.data_root / "*-cql_*.pth")) + glob.glob(str(MODELS_DIR / "*-cql_*.pth"))
                if not candidates:
                    candidates = [str(self.data_root / "cql_policy.pth"), str(MODELS_DIR / "cql_policy.pth")]
                
                candidates = [c for c in candidates if os.path.exists(c)]
                if candidates:
                    candidates.sort(key=os.path.getmtime, reverse=True)
                    policy_path = Path(candidates[0])
                else:
                    policy_path = self.data_root / "cql_policy.pth"
            
            if not policy_path.exists():
                return {"status": "error", "message": "Policy not found"}
                
            basename = policy_path.stem
            cm_path = policy_path.parent / f"{basename}_confusion.png"
            
            if not cm_path.exists():
                return {"status": "error", "message": f"Confusion matrix not found at {cm_path}"}
                
            with open(cm_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                
            return {"status": "success", "image": encoded}
            
        except Exception as e:
            logger.error(f"Failed to load confusion matrix: {e}")
            return {"status": "error", "message": str(e)}


    def _extract_frames(self, ds_path):
        """Extracts frames from camera_0.mp4 into an images/ directory if needed."""
        video_path = ds_path / "camera_0.mp4"
        images_dir = ds_path / "images"
        
        if not video_path.exists():
            return False
            
        if images_dir.exists() and any(images_dir.iterdir()):
            # Already extracted
            return True
            
        logger.debug(f"Extracting frames from {video_path}...")
        images_dir.mkdir(exist_ok=True)
        
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_name = f"frame_{count:06d}.jpg"
            cv2.imwrite(str(images_dir / img_name), frame)
            count += 1
        cap.release()
        logger.debug(f"Extracted {count} frames.")
        return True

    def _parse_sessions(self, dataset_names, extract_goals=True):
        """
        Parses all transitions from selected datasets using the standardized DatasetService.
        """
        from backend.services.datasets import DatasetService
        ds_service = DatasetService(self.data_root)
        
        all_raw_transitions = []
        
        for ds_name in dataset_names:
            ds_path = self.data_root / ds_name
            if not ds_path.exists():
                logger.warning(f"Dataset {ds_name} not found at {ds_path}")
                continue
                
            logger.debug(f"Parsing dataset: {ds_name}")
            self._extract_frames(ds_path) # [NEW] Extract frames to images/ for fast loading without VideoCapture
            session_transitions = ds_service.load_transitions(str(ds_path))
            
            for t in session_transitions:
                t['left_cmd'] = int(t['action'][0])
                t['right_cmd'] = int(t['action'][1])
                
                if 'image_path' in t:
                    p = str(t['image_path']).replace('\\', '/')
                    if len(p) > 2 and p[1] == ':' and p[2] == '/': p = p[2:]
                    if os.path.isabs(p):
                        try: t['image_path'] = os.path.relpath(p, start=self.data_root)
                        except: t['image_path'] = os.path.basename(p)
                            
                if 'video_path' in t:
                    p = str(t['video_path']).replace('\\', '/')
                    if len(p) > 2 and p[1] == ':' and p[2] == '/': p = p[2:]
                    if os.path.isabs(p):
                        try: t['video_path'] = os.path.relpath(p, start=self.data_root)
                        except: t['video_path'] = os.path.basename(p)
            
            all_raw_transitions.extend(session_transitions)
            
        # Sort by global time
        all_raw_transitions.sort(key=lambda t: t.get('timestamp', 0))

        # Goal extraction is now handled directly within VAE training
        # to ensure that goals are uniquely linked to a specific VAE model.
            
        return all_raw_transitions

    def _create_episodes(self, transitions):
        """
        Hindsight relabeling logic specifically tailored for discrete markov datasets.
        Uses a much shorter horizon (e.g. 10 steps max) and relaxes strict stable stop conditions,
        since EVERY step is effectively a stop boundary in the SMDP.
        """
        if not transitions:
            return []
            
        episodes = []
        
        # Group by session
        from itertools import groupby
        transitions_sorted = sorted(transitions, key=lambda x: x['session']) 
        
        for session_name, session_iter in groupby(transitions_sorted, key=lambda x: x['session']):
            session_transitions = list(session_iter)
            if not session_transitions:
                continue
                
            raw_trajectory = session_transitions
            
            if len(raw_trajectory) < MIN_ACTION_STEPS + 1:
                continue
                
            HORIZON = 10
    
            for start_frame_idx in range(len(raw_trajectory)):
                horizon_end = min(start_frame_idx + 1 + HORIZON, len(raw_trajectory))
                possible_goal_indices = list(range(start_frame_idx + 1, horizon_end))
                num_goals_to_sample = min(MAX_GOALS_PER_START_FRAME, len(possible_goal_indices))
                if num_goals_to_sample == 0:
                    continue

                sampled_goal_indices = sorted(random.sample(possible_goal_indices, k=num_goals_to_sample))
                for goal_frame_idx in sampled_goal_indices:
                    start_frame = raw_trajectory[start_frame_idx]
                    goal_frame = raw_trajectory[goal_frame_idx]
                    actions = raw_trajectory[start_frame_idx + 1 : goal_frame_idx + 1]

                    if actions and len(actions) >= MIN_ACTION_STEPS:
                        # [NEW] Deprecate Reverse Action
                        has_reverse = any(a.get('macro_action') == 2 for a in actions)
                        if has_reverse:
                            continue
                            
                        episodes.append({
                            'start_frame': start_frame,
                            'goal_frame': goal_frame,
                            'actions': actions,
                            'action_count': len(actions),
                            'total_frames': len(actions) + 1
                        })
                        
        return episodes
