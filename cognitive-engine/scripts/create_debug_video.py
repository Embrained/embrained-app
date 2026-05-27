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


import cv2
import pandas as pd
import os
import re
import numpy as np

# --- CONFIG ---
# Update this to the valid session path found
SESSION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nook", "capture-2026-01-10 13_12_25")
OUTPUT_VIDEO = "debug_overlay.mp4"
FPS = 10  # Playback speed

def generate_video(session_path):
    log_path = os.path.join(session_path, 'log.csv')
    if not os.path.exists(log_path):
        print(f"Error: No log.csv found in {session_path}")
        return

    # 1. Load Log
    try:
        # The log file has a header: timestamp,img_file,ir,battery,motor_cmd,led_cmd,sound_cmd
        df = pd.read_csv(log_path)
        # Rename columns to match internal logic for convenience, or adjust accessors
        # We need 'img_file' -> 'img', 'motor_cmd' -> 'cmd'
        rename_map = {'img_file': 'img', 'motor_cmd': 'cmd'}
        # Check if columns exist
        if 'img_file' in df.columns and 'motor_cmd' in df.columns:
            df = df.rename(columns=rename_map)
        else:
            print(f"Error: Unexpected columns in log.csv: {df.columns}")
            return

    except Exception as e:
        print(f"Error reading log.csv: {e}")
        return
    
    print(f"Loaded {len(df)} rows.")

    # 2. Check for Duplicates
    # Filter out rows where image filename is missing or NaN
    df = df.dropna(subset=['img'])
    unique_images = df['img'].nunique()
    print(f"Unique Images: {unique_images} | Total Rows: {len(df)}")
    if unique_images < len(df) * 0.5:
        print("WARNING: High duplication detected. Logger is faster than Camera.")

    if len(df) == 0:
        print("Error: Log file is empty or invalid.")
        return

    # 3. Setup Video Writer
    first_img_path = os.path.join(session_path, df.iloc[0]['img'])
    if not os.path.exists(first_img_path):
         print(f"Error: First image not found: {first_img_path}")
         return

    frame = cv2.imread(first_img_path)
    if frame is None:
        print("Error: Could not read first image.")
        return
    
    height, width, layers = frame.shape
    # Use mp4v for compatibility, openh264 often missing on some envs
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

    print(f"Generating video to {OUTPUT_VIDEO}...")
    
    # 4. Loop through Data
    count = 0
    for i, row in df.iterrows():
        img_path = os.path.join(session_path, row['img'])
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Parse Command for Display
        cmd = str(row['cmd'])
        # Regex matches "l: -80; r: 80;" like patterns
        match = re.search(r'l:\s*(-?\d+);\s*r:\s*(-?\d+)', cmd)
        
        status_text = "STOP"
        color = (0, 0, 255) # Red for Stop (BGR)
        
        if match:
            # Note: The logs might have float or int, usually int 0-255 or -100 to 100
            try:
                l, r = int(float(match.group(1))), int(float(match.group(2)))
                
                # Logic: Turn vs Fwd vs Stop
                # Adjust thresholds as per robot (likely > 40 is active)
                if l < -40 and r < -40: # Left (CCW)? Or maybe check polarity
                    # The user said: "Left" Action should result in Positive Flow.
                    # Usually Left turn is Left Motor Back, Right Motor Fwd?
                    # Or both motors one way?
                    # The user's snippet had: if l < -40 and r < -40: status_text = "LEFT"
                    # We'll trust user's snippet logic for labeling
                    status_text = f"LEFT (L:{l} R:{r})"
                    color = (0, 255, 0) # Green
                elif l > 40 and r > 40: # Right (CW)?
                    status_text = f"RIGHT (L:{l} R:{r})"
                    color = (255, 0, 0) # Blue
                elif abs(l) > 40 or abs(r) > 40:
                    status_text = f"MOVING (L:{l} R:{r})"
                    color = (0, 255, 255) # Yellow
            except ValueError:
                pass
            
        # Overlay Text
        # Box background for readability
        # Top bar
        cv2.rectangle(img, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(img, f"Frame: {i} | {status_text}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) # Reduce font size for small 64x64 imgs if needed, but these are raw captures (full res?)
        
        # If image is small (64x64), resizing 0.8 font is huge. 
        # Raw captures are usually larger. Let's check size.
        # If width < 100, resize for display?
        if width < 200:
             img = cv2.resize(img, (width*4, height*4), interpolation=cv2.INTER_NEAREST)
             # Re-draw on larger image
             cv2.rectangle(img, (0, 0), (width*4, 60), (0, 0, 0), -1)
             cv2.putText(img, f"Fr:{i} {status_text}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        video.write(img)
        count += 1

        if count > 500: # Limit to first 500 frames for quick check
            break

    video.release()
    print(f"Done. Saved to {os.path.abspath(OUTPUT_VIDEO)}")

if __name__ == "__main__":
    generate_video(SESSION_PATH)
