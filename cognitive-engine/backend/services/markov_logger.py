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
import cv2
import csv
import time
import queue
import threading
import logging
import json
from datetime import datetime
from config import DATA_DIR


class MarkovLogger:
    """
    Dedicated logger for the Markov controller.
    Logs a single SMDP snapshot whenever a STOP sequence concludes and a MOVE begins.
    Saves JPG images to disk on a background thread to prevent blocking the 10Hz engine.
    """
    def __init__(self, controller_name="none", prefix="markov"):
        # Generate a unique directory name for each recording session
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = os.path.join(DATA_DIR, f'{prefix}_{timestamp_str}')
        self.img_dir = os.path.join(self.base_dir, 'images')
        
        os.makedirs(self.img_dir, exist_ok=True)
        
        # Save metadata.json
        import json
        meta_path = os.path.join(self.base_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({"controller": controller_name}, f, indent=4)
        
        self.csv_path = os.path.join(self.base_dir, 'episode_data.csv')
        self.file_exists = os.path.exists(self.csv_path)
        
        self.write_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        
        self.frame_count = 0
        logging.info(f"MarkovLogger initialized. Directory: {self.base_dir}")

    def log_step(self, frame, ir_raw, batt_raw, ping_raw, motor_str, active_controller="unknown", webcam_frame=None, action_id=None):
        """
        Called EXACTLY once per bout (when STOP transitions to MOVE).
        Sends data to the background thread.
        """
        # Parse motor_str (e.g. "l:130;r:130;")
        pwm_l, pwm_r = 0, 0
        try:
            parts = motor_str.replace("MOCK:", "").strip().split(';')
            for p in parts:
                if 'l:' in p: pwm_l = int(float(p.split(':')[1]))
                if 'r:' in p: pwm_r = int(float(p.split(':')[1]))
        except Exception:
            pass

        timestamp = time.time()
        filename = f"frame_{int(timestamp * 1000)}.jpg"
        
        webcam_filename = None
        if webcam_frame is not None:
             webcam_filename = f"webcam_frame_{int(timestamp * 1000)}.jpg"
             
        row = {
            'timestamp': timestamp,
            'image_file': filename,
            'ir_reading': ir_raw,
            'batt_raw': batt_raw,
            'ping_raw': ping_raw,
            'pwm_left': pwm_l,
            'pwm_right': pwm_r,
            'active_controller': active_controller,
            'action_id': action_id if action_id is not None else ""
        }
        
        self.write_queue.put((frame, filename, row, webcam_frame, webcam_filename))
        self.frame_count += 1

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

    def _writer_loop(self):
        # Open CSV in append mode
        with open(self.csv_path, 'a', newline='') as f:
            fieldnames = ['timestamp', 'image_file', 'ir_reading', 'batt_raw', 'ping_raw', 'pwm_left', 'pwm_right', 'active_controller', 'action_id']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
                
            while self.running or not self.write_queue.empty():
                try:
                    # Block briefly to wait for new frames
                    item = self.write_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                if isinstance(item[0], str) and item[0] == 'GOAL_EVENT':
                    row = item[2]
                    goals_csv_path = os.path.join(self.base_dir, 'goals_data.csv')
                    goals_exists = os.path.exists(goals_csv_path)
                    
                    with open(goals_csv_path, 'a', newline='') as gf:
                        fieldnames = ['timestamp', 'event_type', 'goal_idx', 'manifold_coords', 'steps_to_reach', 'active_model']
                        g_writer = csv.DictWriter(gf, fieldnames=fieldnames)
                        if not goals_exists:
                            g_writer.writeheader()
                        g_writer.writerow(row)  # type: ignore
                    
                    self.write_queue.task_done()
                    continue

                frame, filename, row, webcam_frame, webcam_filename = item
                
                # 1. Save Image
                full_img_path = os.path.join(self.img_dir, filename)
                cv2.imwrite(full_img_path, frame)
                
                # Copy Webcam frame if available
                if webcam_frame is not None and webcam_filename is not None:
                    webcam_full_img_path = os.path.join(self.img_dir, webcam_filename)
                    cv2.imwrite(webcam_full_img_path, webcam_frame)
                
                # 2. Append Metadata
                writer.writerow(row)
                f.flush()
                
                self.write_queue.task_done()

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.info("MarkovLogger closed.")
