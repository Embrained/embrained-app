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

print(f"Loading {EPISODES_PATH}...")
with open(EPISODES_PATH, 'r') as f:
    episodes = json.load(f)

print(f"Loaded {len(episodes)} episodes.")

missing = 0
found = 0
checked = 0

for i, ep in enumerate(episodes[:100]): # Check first 100 episodes
    # Check start frame
    p = os.path.join(DATA_ROOT, ep['start_frame']['image_path'])
    if os.path.exists(p): found += 1
    else: missing += 1
    checked += 1
    
    # Check goal frame
    p = os.path.join(DATA_ROOT, ep['goal_frame']['image_path'])
    if os.path.exists(p): found += 1
    else: missing += 1
    checked += 1

print(f"Checked {checked} images.")
print(f"Found: {found}")
print(f"Missing: {missing}")

if missing > 0:
    print("EXAMPLE MISSING PATH:")
    # Print first missing
    for i, ep in enumerate(episodes[:100]):
         p = os.path.join(DATA_ROOT, ep['start_frame']['image_path'])
         if not os.path.exists(p):
             print(p)
             break
