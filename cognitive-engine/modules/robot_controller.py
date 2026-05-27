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

import json
import os
import logging

class RobotController:
    def __init__(self, config_path="robot_config.json"):
        self.config_path = config_path
        self.deadband_threshold = 0.4
        self.max_safe_speed = 0.8
        self.deadband_threshold = 0.4
        self.max_safe_speed = 0.8
        self.linear_slew_limit = 2.0
        self.angular_slew_limit = 4.0
        
        
        self._load_config()

    def _load_config(self):
        """Load robot configuration from JSON file."""
        if not os.path.exists(self.config_path):
            logging.warning(f"Config file {self.config_path} not found. Using defaults.")
            return

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Extract kinematics settings
            kinematics = config.get("kinematics", {})
            linear = kinematics.get("linear", {})
            angular = kinematics.get("angular", {})
            
            self.deadband_threshold = linear.get("deadband_threshold", self.deadband_threshold)
            self.max_safe_speed = linear.get("max_safe_speed", self.max_safe_speed)
            self.linear_slew_limit = linear.get("slew_limit", self.linear_slew_limit)
            self.angular_slew_limit = angular.get("slew_limit", self.angular_slew_limit)
            
            logging.debug(f"RobotController: Config loaded. Slew Limits: L={self.linear_slew_limit}, A={self.angular_slew_limit}")
        except Exception as e:
            logging.error(f"RobotController: Failed to load config: {e}")


    def process_action(self, action_value):
        """
        Normalize raw action value (-1.0 to 1.0) to PWM duty cycle (0.0 to 1.0)
        with deadband and max safe speed mapping.
        """
        abs_val = abs(action_value)

        # [NEW] PWM Bypass: If value > 1.1, treat as raw PWM integer
        if abs_val > 1.1:
            return int(action_value)

        # Deadband Gate (Normalized Scale)
        if abs_val < 0.01:
            return 0.0
            
        # Remapping: [0.05, 1.0] -> [deadband_threshold, max_safe_speed]
        # Formula: output = deadband + (abs(action) * (max_safe - deadband))
        # Note: The user's formula output = deadband + (abs(action) * (max_safe - deadband))
        # Maps 1.0 to max_safe, but doesn't explicitly account for the 0.05 floor in the mapping itself
        # except as a gate. We follow the user's provided formula exactly.
        
        output = self.deadband_threshold + (abs_val * (self.max_safe_speed - self.deadband_threshold))
        
        # Scale to 8-bit PWM (0-255)
        pwm_val = int(output * 255)
        
        # Preserve Sign
        if action_value < 0:
            pwm_val = -pwm_val
            
        return pwm_val

    def get_motor_commands(self, v, w, quirks=False):
        """
        Mix linear (v) and angular (w) commands into left and right motor PWM values.
        v, w can be normalized [-1.0, 1.0] or raw PWM integers.
        """
        S = self.process_action(v)
        T = self.process_action(w)
        
        if quirks:
            # Legacy Inverted Mapping
            l_val = -S - T
            r_val = S - T
        else:
            # Standard Mapping Logic (Standard convention: CCW/Left is positive)
            l_val = S - T
            r_val = S + T
            
        # Clamp to 8-bit PWM range [-255, 255]
        l_val = max(-255, min(255, int(l_val)))
        r_val = max(-255, min(255, int(r_val)))
        
        return l_val, r_val

if __name__ == "__main__":
    # Test logic
    logging.basicConfig(level=logging.INFO)
    rc = RobotController()
    test_vals = [0.0, 0.04, 0.05, 0.5, 1.0, -0.5, -1.0]
    for v in test_vals:
        print(f"Input: {v:5.2f} -> Output: {rc.process_action(v):5.2f}")
