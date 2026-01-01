
import os
import time
import csv
import logging
import cv2
import numpy as np
from config import DATA_DIR

class DataLogger:
    def __init__(self, run_id=None):
        if run_id is None:
            # Format: capture-YYYY-MM-DD HH_MM_SS
            run_id = time.strftime("capture-%Y-%m-%d %H_%M_%S")
        
        # Save to ./data/datetime/
        self.run_dir = os.path.join(DATA_DIR, run_id)
        # Save to ./data/datetime/
        self.run_dir = os.path.join(DATA_DIR, run_id)
        self.img_dir = self.run_dir
        os.makedirs(self.img_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.run_dir, 'log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # New Schema
        self.writer.writerow(['timestamp', 'img_file', 'ir', 'battery', 'motor_cmd', 'led_cmd', 'sound_cmd'])
        
        logging.info(f"Recording started at {self.run_dir}")

    def log_step(self, frame, ir, battery, motor_cmd, led_cmd, sound_cmd):
        """
        Log a single step.
        """
        ts = time.time()
        # Format: YYYY-MM-DD HH_MM_SS-timestamp_ms.jpg
        # e.g., 2025-06-07 06_08_30-1749334110687.jpg
        ts_ms = int(ts * 1000)
        dt_str = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime(ts))
        img_filename = f"{dt_str}-{ts_ms}.jpg"
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Save Image
        try:
            if frame is not None:
                # Expecting BGR numpy array
                cv2.imwrite(img_path, frame)
        except Exception as e:
            logging.error(f"Failed to save image {img_filename}: {e}")
            
        # Write CSV
        self.writer.writerow([ts, img_filename, ir, battery, motor_cmd, led_cmd, sound_cmd])
        self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        logging.info("Recording saved.")
