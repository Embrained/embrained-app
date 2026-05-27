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
import logging
import json
from config import DATA_DIR, MODELS_DIR

class ModelManager:
    """Handles model discovery and sidecar file management."""
    
    def __init__(self):
        pass

    def find_best_model(self, model_filename):
        """
        Locates the best matching model file.
        Prioritizes:
        1. Exact match in DATA_DIR (recursive)
        2. Exact match in MODELS_DIR
        3. Prefix/Suffix variants (*_model_filename)
        """
        # 1. Exact Match Logic
        search_pattern = os.path.join(DATA_DIR, "**", model_filename)
        candidates = glob.glob(search_pattern, recursive=True)
        
        default_loc = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(default_loc):
            candidates.append(default_loc)
            
        # 2. Variant Logic
        additional_patterns = []
        if "vae_encoder" in model_filename or "tiny_vae" in model_filename:
            additional_patterns.append(f"*_{model_filename}")
            additional_patterns.append("*-vae.pth")
            
        if "cql_policy" in model_filename:
            additional_patterns.append(f"*_{model_filename}")
            additional_patterns.append("*-cql.pth")
            
        for pat in additional_patterns:
            candidates.extend(glob.glob(os.path.join(DATA_DIR, "**", pat), recursive=True))
            candidates.extend(glob.glob(os.path.join(MODELS_DIR, pat)))

        if not candidates:
            return None
            
        # Sort by modification time (Newest First)
        candidates.sort(key=os.path.getmtime, reverse=True)
        
        best = candidates[0]
        logging.info(f"Auto-Discovery: Found {model_filename} (or variant) at {best}")
        return best

    def save_model_goals(self, model_path, image_paths):
        """Save list of goal image paths to a sidecar JSON."""
        if not model_path or not image_paths: return
        try:
            sidecar_path = model_path + ".goals.json"
            with open(sidecar_path, 'w') as f:
                json.dump(image_paths, f)
            logging.info(f"Saved {len(image_paths)} affiliated goals to {sidecar_path}")
        except Exception as e:
            logging.error(f"Failed to save model goals: {e}")

    def load_model_goals(self, model_path):
        """Load list of goal image paths from sidecar JSON."""
        if not model_path: return []
        sidecar_path = model_path + ".goals.json"
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    paths = json.load(f)
                logging.info(f"Loaded {len(paths)} affiliated goals from {sidecar_path}")
                return paths
            except Exception as e:
                logging.error(f"Failed to load model goals: {e}")
        return []

    def infer_dataset_from_model(self, model_path, model_name=None):
        """
        Attempts to infer the dataset directory from a model path.
        Returns absolute path to dataset or None.
        """
        if not model_path: return None
        
        # 1. Check parent directory structure
        if "data" in model_path:
             parent = os.path.dirname(model_path)
             # If directly in dataset folder (e.g. data/Nook/nook.pth)
             dataset_check_files = ["log.csv", "episode_data.csv"]
             if os.path.basename(parent) != "models" and any(os.path.exists(os.path.join(parent, f)) for f in dataset_check_files):
                 return parent
        
        # 2. Name Heuristic
        name = model_name or os.path.basename(model_path)
        parts = name.split('-vae')
        if len(parts) > 0:
            ds_name = parts[0]
            candidate = os.path.join(DATA_DIR, ds_name)
            if os.path.exists(candidate):
                return candidate
                
        return None
