
import time
import threading
import logging
import torch
import numpy as np
import cv2
import base64
import glob
import os
import asyncio

from modules.comms import NervousSystem
# from modules.simulator import Simulator
from modules.vision import VisionSystem
from modules.planner import Planner
from modules.logger import DataLogger
from modules.exploration import ExplorationSystem
from modules.exploration import ExplorationSystem
from backend.manifold import ManifoldService
try:
    from backend.server import AsyncPolicyServer
except ImportError:
    AsyncPolicyServer = None
# Game Modules
from game.referee import MotionReferee
from config import CONTROL_FREQ, GOAL_LED_COLORS, DATA_DIR, MODELS_DIR, ACTION_NAMES, COLOR_NAME_MAP, STOP_DISTANCE_THRESHOLD
import collections

class CognitiveEngine:
    def __init__(self, dry_run=False, use_webcam=False, simulation=False, robot_ip=None, stream_port=81):
        self.dry_run = dry_run
        self.drive_mode = False # Default off, enabled by UI mode switch
        self.use_webcam = use_webcam # Store this for late init
        self.simulation = simulation
        self.robot_ip = robot_ip
        self.stream_port = stream_port
        self.running = False
        self.thread = None
        
        # State Container for UI
        self.state = {
            "image": None,         # Base64 string
            "goal_image": None,    # Base64 string
            "action": "STOP",
            "distance": 0.0,
            "goal_idx": 0,
            "led_color": "N/A",
            "sensor_dist": "0",
            "sensor_batt": "0",
            "mode": "IDLE", # Default to IDLE (Feeds Off)
            "controller": None, # [NEW] Active Explorer or Model
            "fps": 0.0,
            "bvae_model": "N/A",
            "cql_model": "N/A",
            "current_latent": [],
            "goal_latent": [],
            "manifold_coord": None, # [NEW]
            "is_recording": False
        }
        self.state_lock = threading.Lock()
        
        # Live Mode Persistent State
        self.current_live_action = 3 # Default STOP
        self.active_model_name = None # [NEW] Track active trained model
        
        # Command Queue for manual control
        self.command_queue = []
        self.queue_lock = threading.Lock()
        
        # Recording State
        self.recording = False
        self.current_sound_cmd = "s:0;" # Default silence
        self.logger = None # Instantiated on record start

        # State Tracking (Prevent Flooding)
        self.last_led_color = (0, 0, 0)
        self.last_sound_freq = 0

        # VLA State [NEW]
        self.policy_server = None
        self.action_queue = collections.deque()
        self.is_fetching_chunk = False
        self.fetch_lock = threading.Lock()
        
        # Game State
        self.referee = None
        self.game_mode = "GREEN_LIGHT" # or RED_LIGHT
        
        # Initialize Modules
        logging.info("Initializing Cognitive Engine...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # self.comms initialized on demand in Live Mode
            self.comms = None 
            
            # Auto-Discover Models
            enc_path = self._find_best_model("spatial_encoder.pth")
            pol_path = self._find_best_model("cql_policy.pth")
            
            self.vision = VisionSystem(device=self.device, model_path=enc_path)
            self.planner = Planner(device=self.device, model_path=pol_path)
            
            self.explorer = ExplorationSystem() # [NEW]
            self.manifold = ManifoldService(self.vision) # [NEW]
            self.manifold.start_background_fit()
            self.explorer = ExplorationSystem() # [NEW]
            self.manifold = ManifoldService(self.vision) # [NEW]
            self.manifold.start_background_fit()
            
            # VLA Init
            if AsyncPolicyServer:
                self.policy_server = AsyncPolicyServer()
            
            self.referee = MotionReferee() # [NEW]
                
            # self.logger = DataLogger() # Removed auto-init
        except Exception as e:
            logging.critical(f"Engine Init Failed: {e}")
            raise e
            
        # Load Goals
        self.goal_imgs_b64 = self._load_goal_images_b64()
        
        # Stats Tracking
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_frames": 0,
            "actions": {0: 0, 1: 0, 2: 0, 3: 0},
            "action_names": ACTION_NAMES
        }
        
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

        return images

    def _find_best_model(self, model_filename):
        """
        Search for the most recently modified model file in DATA_DIR and its subdirectories.
        """
        search_pattern = os.path.join(DATA_DIR, "**", model_filename)
        candidates = glob.glob(search_pattern, recursive=True)
        
        # Also check default MODELS_DIR
        default_loc = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(default_loc):
            candidates.append(default_loc)
            
        if not candidates:
            return None
            
        # Sort by modification time (descending)
        candidates.sort(key=os.path.getmtime, reverse=True)
        
        best = candidates[0]
        logging.info(f"Auto-Discovery: Found {model_filename} at {best}")
        return best

    def start_feeds(self):
        """Initialize and start camera/robot feeds."""
        if hasattr(self, 'comms') and self.comms is not None:
            return # Already started
            
        logging.info("Starting Feeds (Camera/Serial)...")
        try:
            if self.simulation:
                from modules.simulator import Simulator
                self.comms = Simulator()
                logging.info("Using SIMULATOR for Communication/Sensing")
            else:
                self.comms = NervousSystem(dummy_mode=self.dry_run, use_webcam=self.use_webcam, robot_ip=self.robot_ip, stream_port=self.stream_port)
            logging.info("Feeds Started.")
        except Exception as e:
            logging.error(f"Failed to start feeds: {e}")
            self.comms = None

    def stop_feeds(self):
        """Stop and cleanup camera/robot feeds."""
        if hasattr(self, 'comms') and self.comms is not None:
            logging.info("Stopping Feeds...")
            try:
                # Send STOP before closing
                self.comms.send_command(3)
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
        logging.info("Cognitive Engine Started.")

    def set_mode(self, mode):
        """Switch between LIVE, INFERENCE, TRAINING"""
        # Feed Control Logic
        if mode == 'LIVE':
             self.start_feeds()
        else:
             self.stop_feeds()

        with self.state_lock:
            self.state['mode'] = mode
            if mode == 'LIVE':
                 # Ensure we stop autonomous drive when switching to live
                 self.drive_mode = False 
                 self.current_live_action = 3 # Reset manual state
            elif mode == 'INFERENCE':
                 # Deprecated mode, but kept for compatibility for now logic-wise
                 self.drive_mode = True
            
        logging.info(f"Mode switched to: {mode}")

    def handle_command(self, cmd_type, payload):
        """
        Handle external commands (WS).
        cmd_type: 'MOVE', 'LED', 'SOUND'
        payload: dict or val
        """
        with self.queue_lock:
            self.command_queue.append((cmd_type, payload))

    def start_recording(self):
        if self.recording: return
        try:
            self.logger = DataLogger() # Creates new timestamped dir
            self.recording = True
            logging.info("Recording STARTED.")
        except Exception as e:
            logging.error(f"Failed to start recording: {e}")

    def stop_recording(self):
        if not self.recording: return
        self.recording = False
        if self.logger:
            self.logger.close()
            self.logger = None
        logging.info("Recording STOPPED.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Cleanup
        if hasattr(self, 'comms') and self.comms is not None:
            # Always force STOP on shutdown
            self.comms.send_command(3) # STOP
            self.comms.set_led((0,0,0))
            self.comms.send_sound_command(0)
            self.comms.close()
        
        self.stop_recording() # Ensure recording stops
        
        # if hasattr(self, 'logger'): self.logger.close() # Handled by stop_recording
        
        self.stats['end_time'] = time.time()
        self._print_summary()
        logging.info("Cognitive Engine Stopped.")

    def _print_summary(self):
        duration = self.stats['end_time'] - self.stats['start_time']
        frames = self.stats['total_frames']
        fps = frames / duration if duration > 0 else 0
        
        print("\n" + "="*40)
        print("  EMBRAINED ENGINE SUMMARY STATS")
        print("="*40)
        print(f"  Duration      : {duration:.2f} seconds")
        print(f"  Total Frames  : {frames}")
        print(f"  Avg FPS       : {fps:.2f}")
        print("-" * 40)
        print("  ACTION DISTRIBUTION:")
        for action_id, count in self.stats['actions'].items():
            name = self.stats['action_names'].get(action_id, f"ID {action_id}")
            pct = (count / frames * 100) if frames > 0 else 0
            print(f"    - {name:<10}: {count} ({pct:.1f}%)")
        print("="*40 + "\n")

    def _reconnect_comms(self):
        """Attempts to reconnect the NervousSystem."""
        logging.warning("Reconnecting communication system...")
        try:
             if hasattr(self, 'comms') and self.comms:
                  use_cam = self.comms.use_webcam
                  self.comms.close()
             else:
                  use_cam = self.use_webcam
        except: 
             use_cam = False
        
        try:
             time.sleep(2.0) # Wait for socket cleanup
             if self.simulation:
                 from modules.simulator import Simulator
                 self.comms = Simulator()
                 logging.info("Simulator Reconnected.")
             else:
                 self.comms = NervousSystem(dummy_mode=self.dry_run, use_webcam=use_cam, robot_ip=self.robot_ip, stream_port=self.stream_port)
                 logging.info("Communication System Reconnected.")
        except Exception as e:
             logging.error(f"Reconnection Failed: {e}")

    def _sense(self):
        """Capture and process the latest frame."""
        if not hasattr(self, 'comms') or self.comms is None:
             # If feeds are off (not Live), return None or black
             if self.dry_run or self.state['mode'] != 'LIVE':
                 img = np.zeros((120, 160, 3), dtype=np.uint8)
                 return img, None # No latent if no frame really
             return None, None
             
        frame_bytes = self.comms.get_latest_frame()
        
        if frame_bytes is None and not self.dry_run:
            return None, None

        # Mock frame for dry run if no frame
        frame_to_process = frame_bytes
        if self.dry_run and frame_bytes is None:
             frame_to_process = np.zeros((120, 160, 3), dtype=np.uint8)
        
        img, z_cur = self.vision.process_frame(frame_to_process)
        return img, z_cur

        return img, z_cur

    def _decide(self, current_mode, z_cur, img=None):
        """Decide the next action based on the current mode."""
        target_action = 3 # Default STOP
        dist = 0.0
        goal_idx = 0

        # --- PROCESS COMMAND QUEUE ---
        latest_move = None
        manual_override = False
        
        with self.queue_lock:
            while self.command_queue:
                c_type, c_load = self.command_queue.pop(0)
                if c_type == 'MOVE':
                    latest_move = c_load
                    manual_override = True # Any manual move disables autonomous
                elif c_type == 'SET_CONTROLLER':
                    # Check if it's a trained model or an explorer
                    cmd_name = c_load
                    if cmd_name and cmd_name.endswith('.pth'):
                        logging.info(f"Switching to Model Controller: {cmd_name}")
                        # Try to find model
                        paths = [
                            os.path.join(DATA_DIR, cmd_name),
                            os.path.join(MODELS_DIR, cmd_name),
                            cmd_name # Absolute?
                        ]
                        for p in paths:
                            if os.path.exists(p):
                                if self.planner.load_model(p):
                                    self.active_model_name = cmd_name
                                    self.explorer.set_algorithm(None)
                                    break
                    else:
                        # Explorer or None
                        self.explorer.set_algorithm(cmd_name)
                        self.active_model_name = None
                    
                    # Also reset manual state
                    self.current_live_action = 3 
                elif c_type == 'LED':
                        # Immediate LED
                        if hasattr(self, 'comms') and self.comms:
                            # Check if changed
                            if c_load != self.last_led_color:
                                self.comms.set_led(c_load)
                                self.last_led_color = c_load
                                
                                # Update State for UI
                                r, g, b = c_load
                                color_key = (r, g, b)
                                color_name = COLOR_NAME_MAP.get(color_key, f"rgb({r},{g},{b})")
                                with self.state_lock:
                                    self.state['led_color'] = color_name
                elif c_type == 'SOUND':
                    if hasattr(self, 'comms') and self.comms:
                        if c_load != self.last_sound_freq:
                            self.comms.send_sound_command(c_load)
                            self.last_sound_freq = c_load
        
        # 1. Handle Manual Input priority
        if latest_move is not None:
            # If user pressed a key, we switch off explorer AND model
            if manual_override:
                 self.explorer.set_algorithm(None)
                 self.active_model_name = None
                 
            if isinstance(latest_move, int):
                self.current_live_action = latest_move
            elif isinstance(latest_move, str) and latest_move.isdigit():
                self.current_live_action = int(latest_move)
            else:
                # String mapping
                rev_map = {v: k for k, v in ACTION_NAMES.items()}
                self.current_live_action = rev_map.get(latest_move, 3)
            
            # Refinement: Stop Everything on STOP
            if self.current_live_action == 3:
                # Clear VLA Queue on Stop
                with self.queue_lock:
                    self.action_queue.clear()

                if hasattr(self, 'comms') and self.comms:
                    # Only reset if not already reset
                    if self.last_led_color != (0, 0, 0):
                        self.comms.set_led((0,0,0))
                        self.last_led_color = (0, 0, 0)
                        # Reset UI LED State
                        with self.state_lock:
                            self.state['led_color'] = 'N/A'
                    
                    if self.last_sound_freq != 0:
                        self.comms.send_sound_command(0)
                        self.last_sound_freq = 0

        # 2. Determine Final Action
        if self.active_model_name:
             # Trained Policy Driving
             action, dist, goal_idx = self.planner.decide(z_cur, dist_threshold=STOP_DISTANCE_THRESHOLD)
             target_action = action
        # 2. Determine Final Action
        if self.active_model_name:
             # Trained Policy Driving
             action, dist, goal_idx = self.planner.decide(z_cur, dist_threshold=STOP_DISTANCE_THRESHOLD)
             target_action = action
        elif self.policy_server and current_mode == 'INFERENCE':
             # VLA Driving (Async Chunked)
             target_action = self._get_vla_action(img)
        elif self.explorer.current_algo:
             # Explorer Driving
             target_action = self.explorer.get_action()
        else:
             # Manual state
             target_action = self.current_live_action

        # UI State Update for Controller Name
        # We want to show whichever is active
        active_ctrl = self.active_model_name if self.active_model_name else self.explorer.current_algo
        
        with self.state_lock:
             self.state['controller'] = active_ctrl

        return target_action, dist, goal_idx

    def _act(self, current_mode, target_action, last_action_sent, last_sent_time):
        """Execute the decided action."""
        # Determine efficient command sending (Heartbeat + On-Change)
        # In Inference, we drive if drive_mode is true.
        # In Live, we always drive based on user input (target_action).
        
        should_drive = (current_mode == 'INFERENCE' and self.drive_mode) or (current_mode == 'LIVE')
        effective_action = target_action if should_drive else 3
        
        should_send = (current_mode in ['LIVE', 'INFERENCE']) and \
                      ((effective_action != last_action_sent) or \
                       (time.time() - last_sent_time > 0.5))
        
        motor_cmd_str = "l:0;r:0;"
        new_last_action = last_action_sent
        new_last_sent_time = last_sent_time

        if should_send and hasattr(self, 'comms') and self.comms:
            res = self.comms.send_command(effective_action)
            if res is None:
                raise ConnectionResetError("send_command returned None (Connection Lost)")
            
            motor_cmd_str = res
            new_last_action = effective_action
            new_last_sent_time = time.time()

        return effective_action, motor_cmd_str, new_last_action, new_last_sent_time
    
    def _update_state(self, img, z_cur, effective_action, dist, goal_idx, last_goal_idx, motor_cmd_str):
        """Update shared state and logs."""
        # --- STATS ---
        self.stats['total_frames'] += 1
        self.stats['actions'][effective_action] = self.stats['actions'].get(effective_action, 0) + 1
        
        # --- REFLECT (Update State) ---
        with self.state_lock:
             # Encode Current View
             _, buffer = cv2.imencode('.jpg', img)
             self.state['image'] = base64.b64encode(buffer).decode('utf-8')
             
             # Handle Goal Switch (Autonomous only usually, but let's keep logic)
             led_cmd_str = "d:N/A;"
             new_last_goal_idx = last_goal_idx
             
             if goal_idx != last_goal_idx:
                 new_last_goal_idx = goal_idx
                 if GOAL_LED_COLORS and goal_idx < len(GOAL_LED_COLORS):
                     color = GOAL_LED_COLORS[goal_idx]
                     if color != self.last_led_color and hasattr(self, 'comms') and self.comms:
                         led_cmd_str = self.comms.set_led(color)
                         self.last_led_color = color
            
             # --- RECORDING LOGIC ---
             if self.recording and self.logger:
                 if hasattr(self, 'comms') and self.comms and hasattr(self.comms, 'telemetry'):
                      ir_val = self.comms.telemetry.get('dist', '0')
                      batt_val = self.comms.telemetry.get('batt', '0')
                 else:
                      ir_val = '0'
                      batt_val = '0'
                 
                 self.logger.log_step(
                     img, 
                     ir_val, 
                     batt_val, 
                     motor_cmd_str, 
                     led_cmd_str, 
                     self.current_sound_cmd
                 )
                 
             # Update Goal Image (Static list lookup)
             if goal_idx < len(self.goal_imgs_b64):
                 self.state['goal_image'] = self.goal_imgs_b64[goal_idx]
                 
             # Update Meta
             self.state['action'] = ACTION_NAMES.get(effective_action, '?')
             self.state['distance'] = float(dist)
             self.state['goal_idx'] = goal_idx
             
             self.state['bvae_model'] = self.vision.model_name
             self.state['cql_model'] = self.planner.model_name
             self.state['current_latent'] = z_cur.tolist()
             self.state['is_recording'] = self.recording
             
             if hasattr(self.planner, 'goals') and goal_idx < len(self.planner.goals):
                 self.state['goal_latent'] = self.planner.goals[goal_idx].tolist()
             
             # Project to Manifold
             if self.manifold and self.manifold.is_ready:
                 self.state['manifold_coord'] = self.manifold.project(z_cur)
             
             if self.state['mode'] != 'LIVE':
                 if GOAL_LED_COLORS and goal_idx < len(GOAL_LED_COLORS):
                     self.state['led_color'] = COLOR_NAME_MAP.get(GOAL_LED_COLORS[goal_idx], "Unknown")
            
             # --- TELEMETRY ---
             if hasattr(self, 'comms') and self.comms and hasattr(self.comms, 'telemetry'):
                  self.state['sensor_dist'] = self.comms.telemetry.get('dist', '0')
                  self.state['sensor_batt'] = self.comms.telemetry.get('batt', '0')
        
        return new_last_goal_idx

    def _run_loop(self):
        target_dt = 1.0 / CONTROL_FREQ
        steps = 0
        last_goal_idx = -1
        
        # Stability & Heartbeat
        last_sent_time = time.time()
        last_action_sent = -1
        
        while self.running:
            t_start = time.time()
            
            try:
                # --- SENSE ---
                img, z_cur = self._sense()
                
                # --- OBSERVE ---
                if z_cur is not None:
                    # --- MODE SWITCHING LOGIC ---
                    current_mode = self.state['mode']
                    
                    # --- DECIDE ---
                    target_action, dist, goal_idx = self._decide(current_mode, z_cur, img)
                    
                    # --- ACT ---
                    effective_action, motor_cmd_str, last_action_sent, last_sent_time = self._act(
                        current_mode, target_action, last_action_sent, last_sent_time
                    )
                    
                    # --- REFLECT ---
                    last_goal_idx = self._update_state(
                        img, z_cur, effective_action, dist, goal_idx, last_goal_idx, motor_cmd_str
                    )
            
            except (ConnectionResetError, OSError) as e:
                logging.warning(f"Connection Lost ({e}). Reconnecting...")
                self._reconnect_comms()
            except Exception as e:
                logging.error(f"Unexpected Error in Control Loop: {e}")
                if "WinError 10054" in str(e) or "ConnectionResetError" in str(e):
                     self._reconnect_comms()
                
            steps += 1
            
            # Rate Limiting
            elapsed = time.time() - t_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            self.state['fps'] = 1.0 / (time.time() - t_start)

    def _get_vla_action(self, img):
        """Management of Client-Side Action Queue"""
        # 1. Trigger Fetch if needed (Threshold ~25 actions remaining (0.5s))
        trigger_threshold = 25
        with self.queue_lock:
             current_q_len = len(self.action_queue)
        
        if current_q_len < trigger_threshold and not self.is_fetching_chunk:
             self._trigger_vla_fetch(img)
        
        # 2. Return Action
        if current_q_len > 0:
             with self.queue_lock:
                 # Pop from LEFT (FIFO)
                 # Action format from VLA is likely (dx, dy) or similar
                 # Nimbar expects int (0-3) or string CMD?
                 # app.py's send_command takes int 0-3 (FWD, BWD, LEFT, RIGHT, STOP)
                 # BUT a VLA produces continuous control (v, w) or discrete tokens.
                 # The 'modeling.py' mock produces (50, 2) continuous.
                 # We need to Quantize Continuous -> Discrete for now?
                 # Or update send_command to support continuous "l:x;r:y" strings.
                 # 'comms.send_command' takes payload. 
                 # If we return a string "l:0.5;r:0.5;", comms handles it?
                 # 'send_command' in app logic: "if isinstance(payload, int)... else..."
                 # Let's verify comms in logic or assume we convert to string.
                 
                 chunk_action = self.action_queue.popleft() # Tensor or numpy (2,)
                 
                 # Convert (v, w) to "l:L;r:R;"
                 # Mock conversion:
                 # Assume action is [throttle, turn]
                 # throttle in [-1, 1], turn in [-1, 1]
                 throttle = float(chunk_action[0])
                 turn = float(chunk_action[1])
                 
                 # Differential drive mixing
                 left = throttle + turn
                 right = throttle - turn
                 
                 # Clamp
                 left = max(-1.0, min(1.0, left))
                 right = max(-1.0, min(1.0, right))
                 
                 # "l:%.2f;r:%.2f;"
                 return f"l:{left:.2f};r:{right:.2f};"
        
        return 3 # Stop if starved

    def _trigger_vla_fetch(self, img):
        # Start background thread
        if self.policy_server and img is not None:
             threading.Thread(target=self._run_vla_fetch, args=(img,), daemon=True).start()

    def _run_vla_fetch(self, img):
         # Lock to prevent double fetch
         if self.is_fetching_chunk: return
         
         with self.fetch_lock:
             if self.is_fetching_chunk: return # Double check
             self.is_fetching_chunk = True
             
         try:
             # Prepare Image
             success, encoded_img = cv2.imencode('.jpg', img)
             if not success: return
             
             image_bytes = encoded_img.tobytes()
             
             # Call Async Server (Run in new loop or just run)
             # Note: creating a loop every time is heavy but robust for threads
             actions = asyncio.run(self.policy_server.predict(image_bytes))
             
             # Append to queue
             with self.queue_lock:
                 # "Temporal Ensembling or Replacement"
                 # Instruction says "start with replacement: discard remaining actions"
                 self.action_queue.clear()
                 self.action_queue.extend(actions)
                 
         except Exception as e:
             logging.error(f"VLA Fetch Error: {e}")
         finally:
             with self.fetch_lock:
                 self.is_fetching_chunk = False
