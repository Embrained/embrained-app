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

# Remove the inline imports that cause UnboundLocalError
pattern = re.compile(r"""            # --- DrQ Image Augmentations ---
            import sys
            import os
            sys\.path\.append\(os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)
""", re.DOTALL)
replacement = """            # --- DrQ Image Augmentations ---
"""

text = pattern.sub(replacement, text)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Patch applied to remove local imports.")
