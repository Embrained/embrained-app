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



import math
try:
    import numpy as np
except ImportError:
    np = None

def sanitize_for_json(obj):
    """
    Recursively replace NaN/Infinity with None and convert Numpy types to Python types.
    """
    if obj is None:
        return None

    # Handle floats (standard and numpy)
    if isinstance(obj, float) or (np and isinstance(obj, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    
    # Handle booleans
    if isinstance(obj, bool) or (np and isinstance(obj, np.bool_)):
        return bool(obj)
        
    # Handle integers (standard and numpy)
    if isinstance(obj, int) or (np and isinstance(obj, np.integer)):
        return int(obj)
        
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    
    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
        
    # Handle Numpy Arrays
    if np and isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
        
    return obj

import sys
import subprocess
import logging

_TORCH_CHECKED = False
_TORCH_AVAILABLE = False

def safe_import_torch():
    """
    Safely imports torch, returning None if the import crashes or HANGS.
    """
    global _TORCH_CHECKED, _TORCH_AVAILABLE

    if _TORCH_CHECKED:
        return sys.modules.get('torch', None) if _TORCH_AVAILABLE else None

    if 'torch' in sys.modules:
        _TORCH_AVAILABLE = True
        return sys.modules['torch']
        
    try:
        import torch
        _TORCH_AVAILABLE = True
        return torch
    except Exception as e:
        logging.critical(f"CRITICAL: 'import torch' failed: {e}. Application cannot start without Torch.")
        sys.exit(1)
