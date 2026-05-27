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

DATA_ROOT = r"C:\Users\chris\Embrained\embrained-app\data\nook"
EPISODES_PATH = os.path.join(DATA_ROOT, "episodes.json")

def discretize_action(left, right):
    tol = 40
    if abs(left) < 1 and abs(right) < 1: return 3 # STOP
    if left < -tol and right > tol: return 0 # FWD
    if left < -tol and right < -tol: return 1 # LEFT
    if left > tol and right > tol: return 2 # RIGHT
    if left > tol and right < -tol: return 4 # BACK
    return 3 # Default

print(f"Loading {EPISODES_PATH}...")
with open(EPISODES_PATH, 'r') as f:
    episodes = json.load(f)

print(f"Loaded {len(episodes)} episodes.")

raw_values = []
actions = []

for i, ep in enumerate(episodes[:100]): # Check first 100 episodes
    for step in ep['actions']:
        l = step.get('left_cmd', 0)
        r = step.get('right_cmd', 0)
        raw_values.append((l, r))
        
        act = discretize_action(l, r)
        actions.append(act)

print(f"Checked {len(actions)} steps.")
print(f"Unique Raw Values: {set(raw_values)}")
print(f"Unique Discretized Actions: {set(actions)}")
