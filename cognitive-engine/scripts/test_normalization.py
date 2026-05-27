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
import logging

# Add root to path
sys.path.append(os.getcwd())

from modules.robot_controller import RobotController

def test_normalization():
    logging.basicConfig(level=logging.INFO)
    rc = RobotController(config_path="robot_config.json")
    
    # User's expected mappings:
    # Input 0.0 -> 0.0 (Stop)
    # Input 0.04 -> 0.0 (Deadband)
    # Input 0.05 -> 0.42 (Start of mapping)
    # Input 0.5 -> 0.6 (Halfway)
    # Input 1.0 -> 0.8 (Max)
    
    # Formula check: output = 0.4 + (abs(action) * (0.8 - 0.4))
    # 0.0 -> 0.0
    # 0.04 -> 0.0
    # 0.05 -> 0.4 + (0.05 * 0.4) = 0.4 + 0.02 = 0.42
    # 0.5 -> 0.4 + (0.5 * 0.4) = 0.4 + 0.2 = 0.6
    # 1.0 -> 0.4 + (1.0 * 0.4) = 0.8
    
    test_cases = [
        (0.0, 0.0),
        (0.04, 0.0),
        (0.05, 0.42),
        (0.5, 0.6),
        (1.0, 0.8),
        (-0.5, -0.6),
        (-1.0, -0.8)
    ]
    
    print(f"{'Input':>8} | {'Expected':>10} | {'Actual':>8} | {'Status'}")
    print("-" * 45)
    
    all_passed = True
    for val, expected in test_cases:
        actual = rc.process_action(val)
        status = "PASS" if abs(actual - expected) < 1e-5 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{val:8.2f} | {expected:10.2f} | {actual:8.2f} | {status}")
        
    if all_passed:
        print("\nAll normalization tests PASSED.")
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_normalization()
