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

def find_timestamp(root_dir, target_int):
    print(f"Searching for timestamp starting with {target_int} in {root_dir}...")
    
    target_str = str(target_int)
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'log.csv':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if target_str in line:
                                print(f"FOUND in {file_path} at line {i+1}")
                                #print(f"Content: {line.strip()}")
                                return
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    target_dir = os.path.join(os.getcwd(), 'data', 'livingroom')
    if len(sys.argv) > 1:
        target_val = sys.argv[1]
    else:
        target_val = 1748987309
        
    find_timestamp(target_dir, target_val)
