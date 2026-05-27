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
import random
from glob import glob

# CONFIGURATION
DATASET_NAME = "livingroom"  # What we will call this in the config
DATA_DIR = os.path.join("data", "vint_formatted_livingroom")
OUTPUT_DIR = os.path.join("vint_library", "train", "vint_train", "data", "data_splits", DATASET_NAME)

def main():
    # 1. Get all trajectory folders
    traj_folders = sorted(glob(os.path.join(DATA_DIR, "trajectory_*")))
    traj_names = [os.path.basename(f) for f in traj_folders]
    
    if not traj_names:
        print("No trajectories found! Did you run process_data_vint.py?")
        return

    # 2. Shuffle and Split (90% Train / 10% Test)
    random.seed(42)
    random.shuffle(traj_names)
    
    split_idx = int(len(traj_names) * 0.9)
    train_trajs = traj_names[:split_idx]
    test_trajs = traj_names[split_idx:]
    
    # 3. Write to files
    # The ViNT codebase expects files named 'traj_names.txt' inside 'train' and 'test' folders
    
    for split_name, split_data in [("train", train_trajs), ("test", test_trajs)]:
        split_path = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_path, exist_ok=True)
        
        with open(os.path.join(split_path, "traj_names.txt"), "w") as f:
            for name in split_data:
                f.write(f"{name}\n")
                
        print(f"[{split_name.upper()}] Wrote {len(split_data)} trajectories to {split_path}")

if __name__ == "__main__":
    main()