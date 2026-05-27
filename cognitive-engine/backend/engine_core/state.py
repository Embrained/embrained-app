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


import threading
import collections
from config import STOP_DISTANCE_THRESHOLD, ACTION_NAMES

class StateManager:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Initial State
        self.state = {
            "image": None,         # Base64 string
            "goal_image": None,    # Base64 string
            "action": "STOP",
            "distance": 0.0,
            "stop_threshold": STOP_DISTANCE_THRESHOLD,
            "goal_idx": 0,
            "led_color": "N/A",
            "sensor_dist": "0",
            "sensor_batt": "0",
            "mode": "IDLE", # Default to IDLE (Feeds Off)
            "controller": None,
            "fps": 0.0,
            "bvae_model": "N/A",
            "cql_model": "N/A",
            "base_speed": 0.0,
            "turn_speed": 0.1, # Default placeholder, will be updated by engine
            "current_latent": [],
            "goal_latents": [], 
            "manifold_coord": None, 
            "goal_manifold_coords": [],
            "match_image": None,    
            "match_dist": 0.0,      
            "match_name": "N/A",    
            "is_recording": False,
            "embodiment": "UNKNOWN",
            
            # Training Flags (from routes callbacks)
            "training_epoch": 0,
            "training_loss": 0.0,
            "training_kld": 0.0,
            
            # Paths
            "data_root": "",
            
            "reflex_enabled": False
        }
    
    def update(self, key, value):
        """Thread-safe update of a single key."""
        with self.lock:
            self.state[key] = value
            
    def get(self, key, default=None):
        """Thread-safe get."""
        with self.lock:
            return self.state.get(key, default)
            
    def get_snapshot(self):
        """Thread-safe copy of entire state."""
        with self.lock:
            return self.state.copy()

    def set_mode(self, mode):
        with self.lock:
            self.state['mode'] = mode
            
    def set_led_status(self, color_name):
        with self.lock:
            self.state['led_color'] = color_name
            
    def reset_goals_ui(self):
        with self.lock:
            self.state['goal_idx'] = 0
            self.state['goal_image'] = None
            self.state['goal_latents'] = []
            self.state['goal_manifold_coords'] = []
