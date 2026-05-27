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

print(f"Loading {EPISODES_PATH}...")
with open(EPISODES_PATH, 'r') as f:
    episodes = json.load(f)

lengths = []
for ep in episodes:
    # Length = Start Frame (1) + Actions (N)
    # But train_cql uses full_seq length.
    full_seq_len = 1 + len(ep['actions'])
    lengths.append(full_seq_len)

lengths = np.array(lengths)

print(f"Total Episodes: {len(lengths)}")
print(f"Min Length: {lengths.min()}")
print(f"Max Length: {lengths.max()}")
print(f"Mean Length: {lengths.mean():.2f}")

# Check how many survive the +3 filter
# Logic: for start_idx in range(seq_len - 1): if start_idx + 3 < seq_len: keep
# Essentially, seq_len must be > 1 + 3 = 4?
# idx 0: needs 0+3=3 < seq_len => seq_len > 3 (i.e. 4)
survivors = (lengths > 3).sum()
print(f"Episodes with length > 3 (surviving +3 shift): {survivors} ({survivors/len(lengths)*100:.1f}%)")
