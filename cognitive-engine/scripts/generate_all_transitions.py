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
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.services.datasets import DatasetService

def generate_transitions():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    outputs_path = os.path.join(data_dir, 'all_transitions.json')
    
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and 'markov_' in d]
    if not dirs:
        print(f"No 'markov_' recordings found in {data_dir}.")
        return

    service = DatasetService(data_root=data_dir)
    all_transitions = []
    
    print(f"Aggregating transitions from {len(dirs)} markov sessions...")
    for d in dirs:
        full_path = os.path.join(data_dir, d)
        trans = service.load_transitions(full_path)
        all_transitions.extend(trans)
        print(f"Loaded {len(trans)} from {d}")
        
    with open(outputs_path, 'w') as f:
        json.dump(all_transitions, f)
        
    print(f"\nSuccessfully generated {outputs_path} with {len(all_transitions)} transitions.")

if __name__ == "__main__":
    generate_transitions()
