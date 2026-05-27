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
import numpy as np

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

surviving_actions = []
dropped_actions = []

for ep in episodes:
    seq_len = 1 + len(ep['actions'])
    
    # Logic from train_cql.py
    # We iterate start_idx from 0 to seq_len-2
    for start_idx in range(seq_len - 1):
        
        # Original Action
        act_node = ep['actions'][start_idx] # actions list is 0-indexed aligned with full_seq[1:]
        # Wait, full_seq = [start] + actions.
        # full_seq[0] is start.
        # full_seq[start_idx] is curr_node.
        # If start_idx=0, curr_node=full_seq[0]=Start.
        # The ACTION taken AT start_node is actions[0].
        # So action = actions[start_idx]
        
        l = act_node.get('left_cmd', 0)
        r = act_node.get('right_cmd', 0)
        act_id = discretize_action(l, r)
        
        # Check Survival (200ms / +2 frames)
        if start_idx + 2 >= seq_len:
            dropped_actions.append(act_id)
        else:
            surviving_actions.append(act_id)

print(f"Survivors: {len(surviving_actions)}")
print(f"Dropped: {len(dropped_actions)}")

surv_counts = {i: surviving_actions.count(i) for i in range(5)}
drop_counts = {i: dropped_actions.count(i) for i in range(5)}

print(f"Survivor Distribution: {surv_counts}")
print(f"Dropped Distribution: {drop_counts}")
