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

import logging
import random
import sys
import os

# Add parent directory to path to allow imports from backend/modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.training import TrainingPipeline

# Mock logger to avoid errors
logging.basicConfig(level=logging.INFO)

def test_episode_creation():
    pipeline = TrainingPipeline(".")
    
    # Case 1: Continuous movement (No stops)
    # Expected: 0 episodes with current logic (bug), should be >0 with fix
    print("Testing Case 1: Continuous Movement (No Stops)")
    transitions_no_stop = []
    for i in range(100):
        transitions_no_stop.append({
            'session': 'sess1',
            'timestamp': float(i),
            'image_path': f'sess1/img_{i}.jpg',
            'left_cmd': 100,
            'right_cmd': 100,
            'datetime': '2025-01-01T12:00:00'
        })
        
    episodes = pipeline._create_episodes(transitions_no_stop)
    print(f"Episodes found (No Stops): {len(episodes)}")
    
    # Case 2: Stops at start and end
    # Expected: Should work if >=2 stops
    print("\nTesting Case 2: Stops at Start and End")
    transitions_with_stops = []
    # 2 frames stop
    for i in range(2):
        transitions_with_stops.append({
            'session': 'sess2',
            'timestamp': float(i),
            'image_path': f'sess2/img_{i}.jpg',
            'left_cmd': 0,
            'right_cmd': 0,
            'datetime': '...'
        })
    # Move
    for i in range(2, 50):
        transitions_with_stops.append({
            'session': 'sess2',
            'timestamp': float(i),
            'image_path': f'sess2/img_{i}.jpg',
            'left_cmd': 100,
            'right_cmd': 100,
            'datetime': '...'
        })
    # Stop
    for i in range(50, 52):
        transitions_with_stops.append({
            'session': 'sess2',
            'timestamp': float(i),
            'image_path': f'sess2/img_{i}.jpg',
            'left_cmd': 0,
            'right_cmd': 0,
            'datetime': '...'
        })
        
    episodes = pipeline._create_episodes(transitions_with_stops)
    print(f"Episodes found (With Stops): {len(episodes)}")
    
if __name__ == "__main__":
    test_episode_creation()
