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


import logging
from config import COLOR_NAME_MAP, ACTION_NAMES

class CommandDispatcher:
    def __init__(self, engine):
        self.engine = engine # Reference back to main engine for actions
        
    def dispatch(self, cmd_type, payload):
        """Dispatches commands to specific handlers."""
        if cmd_type == 'MOVE':
            self._handle_move(payload)
        elif cmd_type == 'SET_THRESHOLD':
            self._handle_set_threshold(payload)
        elif cmd_type == 'SET_CONTROLLER':
            self._handle_set_controller(payload)
        elif cmd_type == 'LED':
            self._handle_led(payload)
        elif cmd_type == 'SOUND':
            self._handle_sound(payload)
        elif cmd_type in ('SET_VAE_MODEL', 'SET_BVAE_MODEL'):
            self._handle_set_vae(payload)
        elif cmd_type == 'SET_GOALS':
            self._handle_set_goals(payload)
        elif cmd_type == 'SET_SLAM_MODEL':
            self._handle_set_slam(payload)
        elif cmd_type == 'SET_MANIFOLD_GOAL':
            self._handle_set_manifold_goal(payload)
            
    def _handle_move(self, payload):
        # Notify engine of manual override, UNLESS we are using a human-in-the-loop algorithm
        is_hitl = False
        if self.engine.explorer and self.engine.explorer.current_algo == "MarkovWASD":
            is_hitl = True
            
        action = 0 # STOP
        if isinstance(payload, int):
            action = payload
        elif isinstance(payload, str) and payload.isdigit():
            action = int(payload)
        else:
            # String mapping
            rev_map = {v: k for k, v in ACTION_NAMES.items()}
            action = rev_map.get(payload, 0)
            
        if not is_hitl and action != 0:
            self.engine.set_manual_override(True)
            if getattr(self.engine, 'recording', False):
                logging.info("Manual WASD intervention, auto-stopping recording.")
                self.engine.stop_recording()
            
        self.engine.current_live_action = action
        
        # [NEW] Trigger acceleration logic ONLY on user command receipt
        if self.engine.comms and hasattr(self.engine.comms, 'accelerate_logic'):
            self.engine.comms.accelerate_logic(action)
        
        # Stop specific cleanup logic
        if action == 0:
            if self.engine.vla:
                self.engine.vla.clear_queue()
                
            # Reset hardware state
            if self.engine.comms:
                if self.engine.last_led_color != (0, 0, 0):
                    self.engine.comms.set_led((0,0,0))
                    self.engine.last_led_color = (0, 0, 0)
                    self.engine.state_manager.set_led_status('N/A')
                
                if self.engine.last_sound_freq != 0:
                    self.engine.comms.send_sound_command(0)
                    self.engine.last_sound_freq = 0

    def _handle_set_threshold(self, payload):
        try:
            val = float(payload)
            self.engine.stop_threshold = val
            if hasattr(self.engine, 'explorer') and hasattr(self.engine.explorer, 'neural_oracle'):
                self.engine.explorer.neural_oracle.threshold = val
            logging.info(f"Stop Threshold Updated: {val}")
            self.engine.state_manager.update('stop_threshold', val)
        except ValueError:
            pass

    def _handle_set_controller(self, payload):
        cmd_name = payload
        
        if getattr(self.engine, 'recording', False):
            logging.info("Controller changed, auto-stopping recording.")
            self.engine.stop_recording()
            
        if cmd_name and cmd_name.endswith('.pth'):
            logging.info(f"Switching to Model Controller: {cmd_name}")
            if "_dreamer.pth" in cmd_name:
                self.engine.load_dreamer_model(cmd_name)
            elif "latentslam_" in cmd_name:
                self.engine.load_slam_model(cmd_name)
            else:
                self.engine.load_cql_model(cmd_name)
                

        else:
            # Explorer or None
            self.engine.explorer.set_algorithm(cmd_name)
            
            # [NEW] Synchronize Global VAE strictly for Frontend Manifold plotting
            if isinstance(cmd_name, str) and cmd_name == "Algorithmic Oracle":
                import glob, os
                from config import DATA_DIR
                # Allow all vae models (vqvae, tinyvae, etc)
                vaes = glob.glob(os.path.join(DATA_DIR, "*vae*.pth"))
                # Filter out CQL/Policy weights that share the prefix
                vaes = [v for v in vaes if not any(x in v for x in ['-cql', '_cql', '-fixed_goal', '-oracle', '-markov', '-random_walk', '-e2e'])]
                
                if vaes:
                    vaes.sort(key=os.path.getmtime, reverse=True)
                    latest_vae = os.path.basename(vaes[0])
                    logging.info(f"Auto-syncing global VAE for Algorithmic Oracle Manifold mapping: {latest_vae}")
                    self.engine.load_vae_model(latest_vae)
                    self.engine.state_manager.update('bvae_model', latest_vae)
                    
                    # Also sync a representative target image for the UI inset
                    target_imgs = glob.glob(os.path.join(DATA_DIR, f"{latest_vae.replace('.pth', '')}*.jpg"))
                    if target_imgs:
                        best_img = target_imgs[-1]
                        for img in target_imgs:
                            if 'oracle_control' in img:
                                best_img = img
                                break
                            elif 'image_5' in img:
                                best_img = img
                        logging.info(f"Auto-syncing UI Goal Image and Latents for Algorithmic Oracle: {os.path.basename(best_img)}")
                        try:
                            self.engine._update_runtime_goals([best_img], save_association=False)
                        except Exception as e:
                            logging.error(f"Failed to load UI Goal Image and Latents: {e}")
                    
            self.engine.active_model_name = None
            self.engine.active_model_path = None
            if hasattr(self.engine, 'dreamer_ctrl'):
                self.engine.dreamer_ctrl = None
                
        # [CRITICAL FIX] Reset all Fixed-Goal Evaluation pacing variables to prevent phase leakage
        self.engine.fg_eval_phase = 'MODEL'
        self.engine.fg_eval_bouts = 0
        self.engine.fg_eval_reached = False
        self.engine.fg_eval_step_count = float('nan')
        self.engine.latched_eval_dist = None
                
        # Telemetry Extension Hook (Universal for Oracles AND valid Models)
        if isinstance(cmd_name, str) and (cmd_name in ["Algorithmic Oracle", "Markov"] or cmd_name.endswith('.pth')):
            with self.engine.state_lock:
                is_oracle_req = (cmd_name == "Algorithmic Oracle")
                
                if is_oracle_req:
                    self.engine.telemetry_init_frames = []
                    self.engine.telemetry_warmup_active = True 
                    self.engine.telemetry_target_oracle = True
                    self.engine.telemetry_source_algo = cmd_name if isinstance(cmd_name, str) and "_oracle_control" in cmd_name else "Algorithmic Oracle"
                    self.engine.telemetry_initialized = False
                    self.engine.explorer.set_algorithm("MarkovWASD") # Force warmup pace
                    logging.info(f"Awaiting 10 explicit MarkovWASD transitions to boot {cmd_name} Live Telemetry mapping...")
                else:
                    self.engine.telemetry_warmup_active = False
                    self.engine.telemetry_initialized = False
            if hasattr(self.engine, 'slam_inference'):
                self.engine.slam_inference = None
        
        self.engine.current_live_action = 0

    def _handle_led(self, payload):
        if self.engine.comms:
            if payload != self.engine.last_led_color:
                self.engine.comms.set_led(payload)
                self.engine.last_led_color = payload
                
                r, g, b = payload
                color_key = (r, g, b)
                color_name = COLOR_NAME_MAP.get(color_key, f"rgb({r},{g},{b})")
                self.engine.state_manager.set_led_status(color_name)

    def _handle_sound(self, payload):
        if self.engine.comms:
            if payload != self.engine.last_sound_freq:
                self.engine.comms.send_sound_command(payload)
                self.engine.last_sound_freq = payload

    def _handle_set_vae(self, payload):
        if payload is None:
            self.engine.state_manager.update('bvae_model', "N/A")
            logging.info("VAE Model Deselected")
        else:
            if self.engine.load_vae_model(payload):
                self.engine.state_manager.update('bvae_model', payload)

    def _handle_set_goals(self, payload):
        logging.info(f"Received SET_GOALS Request with {len(payload)} images.")
        self.engine.update_runtime_goals(payload, save_association=True)

    def _handle_set_manifold_goal(self, payload):
        if not isinstance(payload, dict) or 'index' not in payload:
            logging.error(f"Invalid payload for SET_MANIFOLD_GOAL: {payload}")
            return
            
        idx = payload['index']
        if self.engine.manifold and hasattr(self.engine.manifold, 'library_paths'):
            paths = self.engine.manifold.library_paths
            if 0 <= idx < len(paths):
                img_path = paths[idx]
                
                # Cross-Platform Cloud Cache Fallback Resolution
                import os
                from pathlib import Path
                if not os.path.exists(img_path):
                    parts = Path(img_path.replace('\\', '/')).parts
                    if len(parts) >= 3:
                        local_path = os.path.normpath(os.path.join(os.getcwd(), 'data', *parts[-3:]))
                        if os.path.exists(local_path):
                            img_path = local_path
                            
                logging.info(f"Setting manifold goal from index {idx}: {img_path}")
                self.engine.update_runtime_goals([img_path], save_association=True)
            else:
                 logging.error(f"Manifold index {idx} out of bounds (0-{len(paths)-1})")
        else:
             logging.error("Manifold service or library_paths not available")

    def _handle_set_slam(self, payload):
        if payload is None:
            self.engine.slam_inference = None
            logging.info("LatentSLAM Model Deselected")
            self.engine.update_runtime_goals([], save_association=False)
        else:
            self.engine.load_slam_model(payload)
