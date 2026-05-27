import os
import cv2
import csv
import time
import queue
import threading
import logging
import json
from datetime import datetime
from config import DATA_DIR


class ContinuousLogger:
    """
    Dedicated logger for Continuous Action Chunking.
    Logs frames asynchronously as they arrive, independent of motor state.
    """
    def __init__(self, controller_name="continuous", prefix="chunking"):
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = os.path.join(DATA_DIR, f'{prefix}_{timestamp_str}')
        self.img_dir = os.path.join(self.base_dir, 'images')
        
        os.makedirs(self.img_dir, exist_ok=True)
        
        meta_path = os.path.join(self.base_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({"controller": controller_name, "mode": "continuous_action_chunking"}, f, indent=4)
        
        self.csv_path = os.path.join(self.base_dir, 'continuous_data.csv')
        self.file_exists = os.path.exists(self.csv_path)
        
        self.write_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        
        self.frame_count = 0
        logging.info(f"ContinuousLogger initialized. Directory: {self.base_dir}")

    def log_frame(self, frame, ir_raw, batt_raw, ping_raw, executing_action_history, future_chunk, active_controller="unknown", webcam_frame=None):
        """
        Called every time a new frame is processed.
        `executing_action_history`: list of recent PWM tuples that caused this frame.
        `future_chunk`: list of PWM tuples predicted going forward.
        """
        timestamp = time.time()
        filename = f"frame_{int(timestamp * 1000)}.jpg"
        
        webcam_filename = None
        if webcam_frame is not None:
             webcam_filename = f"webcam_frame_{int(timestamp * 1000)}.jpg"
             
        # Serialize chunks to JSON string for CSV compatibility
        history_str = json.dumps(executing_action_history) if executing_action_history else "[]"
        future_str = json.dumps(future_chunk) if future_chunk else "[]"
             
        row = {
            'timestamp': timestamp,
            'image_file': filename,
            'ir_reading': ir_raw,
            'batt_raw': batt_raw,
            'ping_raw': ping_raw,
            'action_history': history_str,
            'future_chunk': future_str,
            'active_controller': active_controller
        }
        
        self.write_queue.put((frame, filename, row, webcam_frame, webcam_filename))
        self.frame_count += 1

    def _writer_loop(self):
        with open(self.csv_path, 'a', newline='') as f:
            fieldnames = ['timestamp', 'image_file', 'ir_reading', 'batt_raw', 'ping_raw', 'action_history', 'future_chunk', 'active_controller']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
                
            while self.running or not self.write_queue.empty():
                try:
                    item = self.write_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                if isinstance(item[0], str) and item[0] == 'GOAL_EVENT':
                    row = item[2]
                    goals_csv_path = os.path.join(self.base_dir, 'goals_data.csv')
                    goals_exists = os.path.exists(goals_csv_path)
                    
                    with open(goals_csv_path, 'a', newline='') as gf:
                        g_fieldnames = ['timestamp', 'event_type', 'goal_idx', 'manifold_coords', 'steps_to_reach', 'active_model']
                        g_writer = csv.DictWriter(gf, fieldnames=g_fieldnames)
                        if not goals_exists:
                            g_writer.writeheader()
                        g_writer.writerow(row)
                    
                    self.write_queue.task_done()
                    continue

                frame, filename, row, webcam_frame, webcam_filename = item
                
                full_img_path = os.path.join(self.img_dir, filename)
                cv2.imwrite(full_img_path, frame)
                
                if webcam_frame is not None and webcam_filename is not None:
                    webcam_full_img_path = os.path.join(self.img_dir, webcam_filename)
                    cv2.imwrite(webcam_full_img_path, webcam_frame)
                
                writer.writerow(row)
                f.flush()
                self.write_queue.task_done()

    def log_goal_event(self, event_type, goal_idx, manifold_coords, steps_taken, active_model=""):
        """
        Logs a goal event (SELECTED, REACHED, ABANDONED) to a separate CSV file.
        """
        timestamp = time.time()
        coords_str = json.dumps(manifold_coords) if manifold_coords is not None else ""
        
        row = {
            'timestamp': timestamp,
            'event_type': event_type,
            'goal_idx': goal_idx,
            'manifold_coords': coords_str,
            'steps_to_reach': steps_taken,
            'active_model': active_model
        }
        self.write_queue.put(('GOAL_EVENT', '', row, None, None))

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.info("ContinuousLogger closed.")
