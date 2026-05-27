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

# Check first episode start frame
start_node = episodes[0]['start_frame']
print(f"Start Frame Keys: {list(start_node.keys())}")
if 'left_cmd' in start_node:
    print(f"Start Frame Left Cmd: {start_node['left_cmd']}")
else:
    print("Start Frame has NO left_cmd")

# Check first action node
action_node = episodes[0]['actions'][0]
print(f"Action 0 Keys: {list(action_node.keys())}")
if 'left_cmd' in action_node:
    print(f"Action 0 Left Cmd: {action_node['left_cmd']}")
