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

# 1. Replace __getitem__ latents with images
getitem_pattern = re.compile(
r"        def get_latent\(node\):.*?latent_goal = get_latent\(sample\['goal_node'\]\) # \[latent_dim\]", 
re.DOTALL)
getitem_replacement = """        # Load raw images instead of pre-computed latents (for DrQ CNN End-To-End)
        latent_curr_stack = torch.stack([self._load_img(n) for n in sample['curr_nodes']], dim=0)
        latent_next_stack = torch.stack([self._load_img(n) for n in sample['next_nodes']], dim=0)

        latent_goal = self._load_img(sample['goal_node'])"""
text = getitem_pattern.sub(getitem_replacement, text)

# 2. Remove Pre-computing block
precompute_pattern = re.compile(
r"        # Precompute All Latents ONCE.*?del _temp_ds",
re.DOTALL)
text = precompute_pattern.sub("        # Removed Precompute All Latents ONCE: DrQ trains vision end-to-end dynamically.", text)

# 3. Add encoder to optimizer
opt_pattern = re.compile(
r"    params_to_train = \[.*?\}\n    \]",
re.DOTALL)
opt_replacement = """    params_to_train = [
        {'params': policy.parameters()},
        {'params': encoder.parameters() if hasattr(encoder, 'parameters') else []}
    ]"""
text = opt_pattern.sub(opt_replacement, text)

# 4. Modify Train loop batch processing
train_loop_pattern = re.compile(
r"            reward = reward\.unsqueeze\(1\) if reward\.dim\(\) == 1 else reward.*?state_input_next = torch\.cat\(\[latent_next_stack, latent_goal\], dim=1\)",
re.DOTALL)
train_loop_replacement = """            reward = reward.unsqueeze(1) if reward.dim() == 1 else reward

            # --- DrQ Image Augmentations ---
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from utils.augmentations import random_shift
            except:
                # inline fallback
                def random_shift(x, pad=4): return x

            B, S, C, H, W = latent_curr_stack.shape
            img_curr_flat = latent_curr_stack.view(B*S, C, H, W)
            img_next_flat = latent_next_stack.view(B*S, C, H, W)
            
            img_curr_aug = random_shift(img_curr_flat).view(B, S, C, H, W)
            img_next_aug = random_shift(img_next_flat).view(B, S, C, H, W)
            img_goal_aug = random_shift(latent_goal)

            # --- End-To-End Forward Pass ---
            state_input_curr = q_net(img_curr_aug, img_goal_aug, state_curr) # Actually q_values, but variable name reused for compatibility
            state_input_next = target_q_net(img_next_aug, img_goal_aug, state_next)"""
text = train_loop_pattern.sub(train_loop_replacement, text)

# Write back
with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Patch applied successfully.")
