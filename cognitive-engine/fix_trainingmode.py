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

import re

with open("frontend/src/components/TrainingMode.jsx", "r", encoding="utf-8") as f:
    text = f.read()

target = """                if (json.status === "success" && json.image) {
                    setVaeValidationImage(`data:image/png;base64,${json.image}`);
                } else {
                    setVaeValidationImage(null);
                }"""

replacement = """                if (json.status === "success" && json.image) {
                    setVaeValidationImage(`data:image/png;base64,${json.image}`);
                } else {
                    console.warn("verifyManifold returned no image, keeping existing.");
                }"""

pattern = re.compile(re.escape(target).replace(r'\n', r'\r?\n'), re.MULTILINE)

if pattern.search(text):
    text = pattern.sub(replacement, text)
    with open("frontend/src/components/TrainingMode.jsx", "w", encoding="utf-8", newline='\n') as f:
        f.write(text)
    print("Patched TrainingMode")
else:
    print("Target not found")
