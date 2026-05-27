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

import pickle
import numpy as np
import os

path = r'c:\Users\chris\Embrained\embrained-app\data\vint_formatted_livingroom\trajectory_0\traj_data.pkl'

if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print("Keys:", data.keys())
    for k, v in data.items():
        if isinstance(v, (np.ndarray, list)):
            print(f"{k}: type={type(v)}, shape={np.array(v).shape if isinstance(v, list) else v.shape}")
        else:
            print(f"{k}: {v}")
