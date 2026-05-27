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
import sys
import time
import cv2
import numpy as np
import logging

# Add current dir to path to import modules
sys.path.append(os.getcwd())

from modules.simulator import Simulator

def verify_sim_rendering():
    print("\n--- VERIFY: Simulator Rendering ---")
    logging.basicConfig(level=logging.INFO)
    
    # Init simulator (Headless=True)
    sim = Simulator(headless=True)
    
    try:
        # Check URDF loads
        if sim.robotId < 0:
            print("[FAIL] Robot failed to load")
            return False
            
        print(f"[OK] Robot ID: {sim.robotId}")
        
        # Take a few frames
        print("Taking test frames...")
        for i in range(5):
            img = sim.get_latest_frame()
            if img is None:
                print(f"[FAIL] Frame {i} is None")
                continue
            
            # Check for non-zero pixels
            avg_val = np.mean(img)
            max_val = np.max(img)
            print(f"Frame {i}: Shape {img.shape}, Mean {avg_val:.2f}, Max {max_val}")
            
            # Check telemetry
            dist = float(sim.telemetry.get('dist', '0'))
            print(f"Telemetry Dist: {dist:.1f} cm")
            
            if max_val > 0:
                print(f"[OK] Frame {i} contains data!")
                cv2.imwrite(f"debug_frame_{i}.jpg", img)
            else:
                print(f"[WARNING] Frame {i} is solid black")
            
            # Move robot slightly
            sim.send_command(0) # Forward
            time.sleep(0.5)
            
        return True

    finally:
        sim.close()
        print("--- End of Verification ---\n")

if __name__ == "__main__":
    verify_sim_rendering()
