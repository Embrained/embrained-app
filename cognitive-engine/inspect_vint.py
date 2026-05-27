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

import inspect
import sys
import os

# Setup path to find the library
sys.path.append(os.path.join(os.getcwd(), "vint_library", "train"))

try:
    from vint_train.data.vint_dataset import ViNT_Dataset
    print("\n=== ViNT_Dataset.__init__ Arguments ===")
    sig = inspect.signature(ViNT_Dataset.__init__)
    print(sig)
    print("=======================================\n")
    
    # Also print the docstring if it exists
    print("Docstring:")
    print(ViNT_Dataset.__init__.__doc__)
    
except Exception as e:
    print(f"Error inspecting: {e}")