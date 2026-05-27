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


import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from modules.simulator import Simulator
    print("SUCCESS: Imported Simulator")
except ImportError as e:
    print(f"FAILURE: Could not import Simulator: {e}")
    sys.exit(1)

def test_sim():
    try:
        sim = Simulator(headless=True)
        print("SUCCESS: Instantiated Simulator")
        
        # Test Frame
        img = sim.get_latest_frame()
        if img is not None and isinstance(img, np.ndarray) and img.shape == (240, 320, 3):
             print(f"SUCCESS: Got Frame {img.shape}")
        else:
             print(f"FAILURE: Invalid Frame: {type(img)}")
        
        # Test Command
        sim.send_command(1) # Fwd
        print("SUCCESS: Sent Forward Command")
        
        time.sleep(1)
        
        sim.send_command(0) # Stop
        print("SUCCESS: Sent Stop Command")
        
        # Test Telemetry
        if hasattr(sim, 'telemetry'):
             print(f"SUCCESS: Telemetry found: {sim.telemetry}")
        else:
             print("FAILURE: No Telemetry")
             
        sim.close()
        print("SUCCESS: Simulation Closed")
        
    except Exception as e:
        print(f"FAILURE: Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sim()
