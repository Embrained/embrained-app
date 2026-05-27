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

manifold_file = r"c:/Users/chris/Embrained/software_suite/analysis_archive/verify_manifold.py"
with open(manifold_file, 'r', encoding='utf-8') as f:
    text = f.read()

target = "return np.array([]), torch.tensor([])"
replacement = "return torch.tensor([])"

if target in text:
    text = text.replace(target, replacement)
    
with open(manifold_file, 'w', encoding='utf-8') as f:
    f.write(text)

print("Done")
