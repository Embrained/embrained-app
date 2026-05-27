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
import json
import random
import sys
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, ACTION_DIM, ACTION_PWM_MAP
import math

data_root = DATA_DIR
trans_path = os.path.join(data_root, "all_transitions.json")
if not os.path.exists(trans_path):
    print("Transitions file not found.")
    sys.exit(1)

with open(trans_path, 'r') as f:
    all_data = json.load(f)

action_names = {0: "STOP", 1: "FWD", 2: "REV", 3: "LEFT", 4: "RIGHT"}
def get_action_name(curr_node):
    if 'macro_action' in curr_node:
        action = int(curr_node['macro_action'])
        if action >= ACTION_DIM: action = 0
    else:
        raw_l = float(curr_node.get('left_cmd', 0.0))
        raw_r = float(curr_node.get('right_cmd', 0.0))
        best_action = 0
        best_dist = float('inf')
        for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
            d = math.hypot(raw_l - map_l, raw_r - map_r)
            if d < best_dist:
                best_dist = d
                best_action = act_id
        action = best_action
    return action_names.get(action, "UNK")

sessions = {}
for item in all_data:
    s = item.get('session', 'default')
    if s not in sessions: sessions[s] = []
    sessions[s].append(item)

trajectories = []
for s in sessions:
    traj = sorted(sessions[s], key=lambda x: x.get('timestamp', 0))
    if len(traj) > 10:
        trajectories.append(traj)

random.seed(80085) # Reproducible
episodes = []
while len(episodes) < 5:
    traj = random.choice(trajectories)
    if len(traj) < 10: continue
    start_idx = random.randint(0, len(traj) - 7)
    episodes.append([traj[i] for i in range(start_idx, start_idx + 6)])

fig, axes = plt.subplots(5, 6, figsize=(18, 15))

for row_idx, ep in enumerate(episodes):
    for col_idx in range(6):
        node = ep[col_idx]
        ax = axes[row_idx, col_idx]
        
        p = node.get('image_path')
        if p and not os.path.isabs(p):
            p = os.path.join(data_root, p)
            
        if p and os.path.exists(p):
            img = Image.open(p).convert('RGB')
            ax.imshow(img)
            
            # Draw borders
            if col_idx == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor('purple')
                    spine.set_linewidth(4)
            elif col_idx == 5:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(4)
        else:
            # Black image
            ax.imshow(Image.new('RGB', (64, 64), (0, 0, 0)))
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        if col_idx == 0:
            act_name = get_action_name(node)
            ax.set_title(f"START\nAction taken: {act_name}", color='purple')
        elif col_idx == 5:
            ax.set_title("GOAL", color='green', fontweight='bold')
        else:
            act_name = get_action_name(node)
            ax.set_title(f"Action taken:\n{act_name}")
            
plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "her_grid.png")
plt.savefig(out_png, dpi=150)
print(f"Grid saved to {out_png}")
