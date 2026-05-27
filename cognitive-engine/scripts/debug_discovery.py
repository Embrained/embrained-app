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
import glob
import sys

# Mock Config
DATA_DIR = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(os.getcwd(), "models")

def find_best_model(model_filename):
    print(f"Searching for: {model_filename}")
    # 1. Exact Match Logic (Current)
    search_pattern = os.path.join(DATA_DIR, "**", model_filename)
    candidates = glob.glob(search_pattern, recursive=True)
    
    default_loc = os.path.join(MODELS_DIR, model_filename)
    if os.path.exists(default_loc):
        candidates.append(default_loc)
        
    # 2. [NEW] Prefix Match Logic
    additional_patterns = []
    if "vae_encoder" in model_filename or "tiny_vae" in model_filename:
        additional_patterns.append(f"*_{model_filename}")
        additional_patterns.append("*-vae.pth")
        
    if "cql_policy" in model_filename:
        additional_patterns.append(f"*_{model_filename}")
        additional_patterns.append("*-cql.pth")
        
    for pat in additional_patterns:
        print(f"Checking Pattern: {pat}")
        candidates.extend(glob.glob(os.path.join(DATA_DIR, "**", pat), recursive=True))
        candidates.extend(glob.glob(os.path.join(MODELS_DIR, pat)))

    if not candidates:
        print("No candidates found.")
        return None
        
    # Sort by modification time (Newest First)
    candidates.sort(key=os.path.getmtime, reverse=True)
    
    best = candidates[0]
    print(f"Found best candidate: {best}")
    return best

if __name__ == "__main__":
    print(f"CWD: {os.getcwd()}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    
    # Test for vae_encoder.pth
    res = find_best_model("vae_encoder.pth")
    if not res:
        print("Trying tiny_vae_final.pth")
        res = find_best_model("tiny_vae_final.pth")
        
    print(f"Final Result: {res}")
