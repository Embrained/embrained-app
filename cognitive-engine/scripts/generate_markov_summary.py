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
import csv
from pathlib import Path

def main():
    data_dir = Path(r"c:\Users\chris\Embrained\software_suite\data")
    summary_file = data_dir / "data_summary.csv"
    
    results = []
    
    # Iterate over all directories starting with "markov_"
    for entry in data_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("markov_"):
            episode_file = entry / "episode_data.csv"
            
            num_transitions = 0
            if episode_file.exists():
                with open(episode_file, "r", encoding="utf-8") as f:
                    # Count non-empty lines minus 1 for header
                    lines = [line for line in f if line.strip()]
                    num_transitions = max(0, len(lines) - 1)
            else:
                # If there's no episode_data.csv, transitions are 0
                pass
                
            results.append([entry.name, num_transitions])
            
    # Sort for consistent output
    results.sort(key=lambda x: x[0])
    
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Directory Name", "Transitions Logged"])
        writer.writerows(results)
        
    print(f"Generated summary at {summary_file}")

if __name__ == "__main__":
    main()
