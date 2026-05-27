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
import re

file_path = "c:/Users/chris/Embrained/software_suite/backend/train_cql.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix the Q-value usage
pattern = re.compile(r"q_values = q_net\.policy\(state_input_curr\).*?next_q_values = target_q_net\.policy\(state_input_next\)", re.DOTALL)
replacement = """q_values = state_input_curr
            q_action = q_values.gather(1, action.view(-1, 1))

            # Target Q
            with torch.no_grad():
                # Double DQN / Standard DQN
                next_q_values = state_input_next"""
text = pattern.sub(replacement, text)

# Disable wall seeking since it relies on deleted pre-computed latents
eval_pattern = re.compile(r"cm_path = evaluate_wall_seeking\(q_net, dataset, trajectories, device\)", re.DOTALL)
text = eval_pattern.sub("cm_path = None # Disabled: wall seeking metric is not compatible with DrQ raw-image inputs yet", text)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Patch 2 applied successfully.")
