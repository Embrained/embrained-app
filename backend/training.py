
import os
import json
import csv
import re
import random
import logging
import threading
from datetime import datetime
from pathlib import Path
# from .get_embeddings import EmbeddingGenerator # Obsolete
from .train_cql import train as train_cql_model

# Logger setup
logger = logging.getLogger("TrainingPipeline")

# Constants
DELAY_FRAMES = 1
MIN_ACTION_STEPS = 2
MAX_GOAL_DISTANCE_HORIZON = 50
MAX_GOALS_PER_START_FRAME = 3
STOP_ACTION = (0, 0)
MOTOR_RE = re.compile(r'l: *(-?\d+);r: *(-?\d+);')

class TrainingPipeline:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.lock = threading.Lock()
        self.is_processing = False
        
    def process_datasets(self, dataset_names):
        """
        Main entry point to process selected datasets.
        """
        with self.lock:
            if self.is_processing:
                return {"status": "error", "message": "Already processing"}
            self.is_processing = True

        try:
            # Step 1: Parse Transitions
            all_transitions = self._parse_sessions(dataset_names)
            
            # Save transitions (intermediate)
            transitions_path = self.data_root / "all_transitions.json"
            with open(transitions_path, 'w') as f:
                json.dump(all_transitions, f, indent=2)
                
            # Step 2: Hindsight Relabeling
            episodes = self._create_episodes(all_transitions)
            
            # Save episodes (final)
            episodes_path = self.data_root / "episodes.json"
            with open(episodes_path, 'w') as f:
                json.dump(episodes, f, indent=2)
                
            return {
                "status": "success", 
                "transitions_count": len(all_transitions),
                "episodes_count": len(episodes)
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            with self.lock:
                self.is_processing = False

    def run_cql_pipeline(self, num_epochs=5, stop_event=None, progress_callback=None):
        """
        Executes the full Spatial-CQL training pipeline:
        1. Train Encoder (Spatial Softmax) + Q-Network (Discrete CQL) Jointly
        """
        with self.lock:
            if self.is_processing:
                return {"status": "error", "message": "Already processing"}
            self.is_processing = True

        try:
            # Step 1: Training (Now includes on-the-fly embedding)
            if progress_callback: progress_callback(0, 0.0)
            logger.info("Starting Spatial-CQL Training...")
            
            # The training function now handles everything using raw images
            model_path_enc, model_path_pol = train_cql_model(
                self.data_root, 
                num_epochs=num_epochs,
                stop_event=stop_event,
                progress_callback=progress_callback
            )
            
            return {
                "status": "success",
                "encoder_path": model_path_enc,
                "policy_path": model_path_pol
            }
            
        except Exception as e:
            logger.error(f"CQL Pipeline failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            with self.lock:
                self.is_processing = False

    def _parse_sessions(self, dataset_names):
        """
        Parses log.csv from each dataset and aligns images.
        """
        all_raw_transitions = []
        
        for ds_name in dataset_names:
            ds_path = self.data_root / ds_name
            log_path = ds_path / "log.csv"
            
            if not log_path.exists():
                logger.warning(f"Log file missing for {ds_name}, skipping.")
                continue
                
            # Read CSV
            # Format: timestamp, img_file, ir, battery, motor_cmd, led_cmd, sound_cmd
            session_transitions = []
            try:
                with open(log_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Extract Motor Cmd
                        m_cmd_str = row.get('motor_cmd', '')
                        m = MOTOR_RE.search(m_cmd_str)
                        if m:
                            l_cmd, r_cmd = map(int, m.groups())
                        else:
                            # Default stop if parse fails or empty
                            l_cmd, r_cmd = 0, 0
                            
                        # Construct Image Path (relative to data root for portability?)
                        # Or relative to the dataset folder?
                        # The user's script used relative to data_dir.
                        # We'll store: dataset_name/images/filename
                        img_filename = row.get('img_file', '')
                        
                        # Determine path structure (New/Legacy vs Old VAE format)
                        if (ds_path / "images").exists():
                            img_rel_path = f"{ds_name}/images/{img_filename}"
                        else:
                            img_rel_path = f"{ds_name}/{img_filename}"
                        
                        # Timestamp
                        ts = float(row.get('timestamp', 0))
                        
                        session_transitions.append({
                            'session': ds_name,
                            'timestamp': ts,
                            'image_path': img_rel_path,
                            'left_cmd_raw': l_cmd,
                            'right_cmd_raw': r_cmd,
                            'datetime': datetime.fromtimestamp(ts).isoformat()
                        })
            except Exception as e:
                logger.error(f"Error parsing log.csv for {ds_name}: {e}")
                continue
                
            all_raw_transitions.extend(session_transitions)
            
        # Sort by global time? Or keep blocked by session?
        # User script sorts by datetime global.
        all_raw_transitions.sort(key=lambda x: x['timestamp'])

        # Apply Delay
        final_transitions = []
        original_cmds = [(t['left_cmd_raw'], t['right_cmd_raw']) for t in all_raw_transitions]
        # Pad with (0,0)
        padded_cmds = [(0, 0)] * DELAY_FRAMES + original_cmds

        for i, record in enumerate(all_raw_transitions):
            delayed_l, delayed_r = padded_cmds[i]
            final_transitions.append({
                'session': record['session'],
                'timestamp': record['timestamp'],
                'image_path': record['image_path'],
                'left_cmd': delayed_l,
                'right_cmd': delayed_r, # User script had -delayed_r_raw, assuming inversion needed? 
                                        # But our engine usually logs logical cmds. 
                                        # Let's stick strictly to log unless user says otherwise.
                                        # Actually, user script said: 'right_cmd': -delayed_r_raw
                                        # Maybe their robot needs that. I will respect the user's provided logic.
                                        # "Use similar logic to this".
                                        # I will stick to exact values from log.csv for now.
                'datetime': record['datetime']
            })
            
        return final_transitions

    def _create_episodes(self, transitions):
        """
        Hindsight relabeling logic.
        """
        if not transitions:
            return []
            
        stable_stop_indices = {
            i for i in range(1, len(transitions))
            if (transitions[i]['left_cmd'], transitions[i]['right_cmd']) == STOP_ACTION
            and (transitions[i-1]['left_cmd'], transitions[i-1]['right_cmd']) == STOP_ACTION
        }
        
        episodes = []
        sorted_stops = sorted(list(stable_stop_indices))
        
        # Add index 0 as start if not consistent? User logic loops through stops.
        # If first frame is not stop, we might miss the first episode?
        # User script: range(len(sorted_stops) - 1) -> implies segments between stops.
        
        for i in range(len(sorted_stops) - 1):
            start_idx = sorted_stops[i]
            next_stop_idx = sorted_stops[i+1]
            raw_trajectory = transitions[start_idx : next_stop_idx + 1]

            if len(raw_trajectory) < MIN_ACTION_STEPS + 1:
                continue

            for start_frame_idx in range(len(raw_trajectory)):
                horizon_end = min(start_frame_idx + 1 + MAX_GOAL_DISTANCE_HORIZON, len(raw_trajectory))
                possible_goal_indices = list(range(start_frame_idx + 1, horizon_end))
                
                num_goals_to_sample = min(MAX_GOALS_PER_START_FRAME, len(possible_goal_indices))
                if num_goals_to_sample == 0:
                    continue
                    
                sampled_goal_indices = random.sample(possible_goal_indices, k=num_goals_to_sample)
                
                for goal_frame_idx in sampled_goal_indices:
                    start_frame = raw_trajectory[start_frame_idx]
                    goal_frame = raw_trajectory[goal_frame_idx]
                    actions = raw_trajectory[start_frame_idx + 1 : goal_frame_idx + 1]
                    
                    if actions and len(actions) >= MIN_ACTION_STEPS:
                        episodes.append({
                            'start_frame': start_frame,
                            'goal_frame': goal_frame,
                            'actions': actions,
                            'action_count': len(actions),
                            'total_frames': len(actions) + 1
                        })
                        
        return episodes
