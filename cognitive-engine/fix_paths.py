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

datasets_file = r"c:/Users/chris/Embrained/software_suite/backend/services/datasets.py"
with open(datasets_file, 'r', encoding='utf-8') as f:
    text = f.read()

target = """    def list_datasets(self, path=None, fast=False):
        \"\"\"List all dataset directories in the specified path or ./data default.\"\"\"
        # Force absolute and normalized
        if path and path.strip():
            # [FIX] Handle potential mixed slashes from frontend
            path = path.replace('\\\\', '/')
            data_dir = os.path.abspath(path) if os.path.isabs(path) else os.path.abspath(os.path.join(self.data_root, path))
        else:
            data_dir = os.path.abspath(self.data_root)"""

replacement = """    def list_datasets(self, path=None, fast=False):
        \"\"\"List all dataset directories in the specified path or ./data default.\"\"\"
        # Force absolute and normalized
        if path and path.strip():
            # [FIX] Handle potential mixed slashes from frontend
            path = path.replace('\\\\', '/')
            
            # [FIX] Cross-OS Path Sanitization
            if len(path) > 2 and path[1] == ':' and path[2] == '/':
                path = path[2:] # Strip Windows drive letter
                
            data_root_fwd = str(self.data_root).replace('\\\\', '/')
            if os.path.isabs(path) and not path.startswith(data_root_fwd):
                parts = path.split('/')
                if "data" in parts:
                    idx = parts.index("data")
                    subpath = "/".join(parts[idx+1:])
                    path = subpath if subpath else ""
                else:
                    path = os.path.basename(path)
                    
            data_dir = os.path.abspath(path) if os.path.isabs(path) else os.path.abspath(os.path.join(self.data_root, path))
        else:
            data_dir = os.path.abspath(self.data_root)"""

if target in text:
    text = text.replace(target, replacement)
elif target.replace("\n", "\r\n") in text:
    text = text.replace(target.replace("\n", "\r\n"), replacement.replace("\n", "\r\n"))
else:
    print("Could not find list_datasets target")

with open(datasets_file, 'w', encoding='utf-8') as f:
    f.write(text)

training_file = r"c:/Users/chris/Embrained/software_suite/backend/training/__init__.py"
with open(training_file, 'r', encoding='utf-8') as f:
    text2 = f.read()

target2 = """            for t in session_transitions:
                t['left_cmd'] = int(t['action'][0])
                t['right_cmd'] = int(t['action'][1])
                if 'image_path' in t and os.path.isabs(t['image_path']):
                    t['image_path'] = os.path.relpath(t['image_path'], start=self.data_root)"""

replacement2 = """            for t in session_transitions:
                t['left_cmd'] = int(t['action'][0])
                t['right_cmd'] = int(t['action'][1])
                
                if 'image_path' in t:
                    p = str(t['image_path']).replace('\\\\', '/')
                    if len(p) > 2 and p[1] == ':' and p[2] == '/': p = p[2:]
                    if os.path.isabs(p):
                        try: t['image_path'] = os.path.relpath(p, start=self.data_root)
                        except: t['image_path'] = os.path.basename(p)
                            
                if 'video_path' in t:
                    p = str(t['video_path']).replace('\\\\', '/')
                    if len(p) > 2 and p[1] == ':' and p[2] == '/': p = p[2:]
                    if os.path.isabs(p):
                        try: t['video_path'] = os.path.relpath(p, start=self.data_root)
                        except: t['video_path'] = os.path.basename(p)"""

if target2 in text2:
    text2 = text2.replace(target2, replacement2)
elif target2.replace("\n", "\r\n") in text2:
    text2 = text2.replace(target2.replace("\n", "\r\n"), replacement2.replace("\n", "\r\n"))
else:
    print("Could not find _parse_sessions target")

with open(training_file, 'w', encoding='utf-8') as f:
    f.write(text2)

print("Done")
