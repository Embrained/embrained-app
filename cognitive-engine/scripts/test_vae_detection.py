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


import torch
import logging
import sys
import os

# Create a logger
logger = logging.getLogger("TestVAEDetection")
logging.basicConfig(level=logging.INFO)

def test_detection():
    print("Testing VAE Model Size Detection Logic...")
    
    # 1. Simulate a "Tiny" model state dict (16 channels)
    # TinyVAE structure: encoder.0 is Conv2d(3, 16, ...)
    # Weight shape: (Out, In, K, K) -> (16, 3, 4, 4)
    tiny_state = {
        'fc_mu.weight': torch.randn(32, 4096), # Latent=32, Flatten=4096 (Ambiguous!)
        'encoder.0.weight': torch.randn(16, 3, 4, 4)
    }
    
    # 2. Simulate a "Medium" model state dict (32 channels)
    # MediumVAE structure: encoder.0 is Conv2d(3, 32, ...)
    # Weight shape: (Out, In, K, K) -> (32, 3, 4, 4)
    medium_state = {
        'fc_mu.weight': torch.randn(32, 4096), # Latent=32, Flatten=4096 (Ambiguous!)
        'encoder.0.weight': torch.randn(32, 3, 4, 4)
    }
    
    # --- Logic from train_cql.py ---
    def run_logic(loaded_state):
        model_size_detected = 'small'
        
        # Standard Detection
        if 'fc_mu.weight' in loaded_state:
            weight_shape = loaded_state['fc_mu.weight'].shape
            flatten_dim = weight_shape[1]
            
            if flatten_dim == 8192:
                model_size_detected = 'small'
            elif flatten_dim == 4096:
                model_size_detected = 'medium' # Default heuristic
            elif flatten_dim == 2048:
                model_size_detected = 'large'
            elif flatten_dim == 1024:
                model_size_detected = 'enormous'
            else:
                logger.warning(f"Unknown VAE flatten dim: {flatten_dim}. Defaulting to small.")
    
        # Refined Detection: Tiny (16) vs Medium (32) ambiguity at 4096 dim
        if model_size_detected == 'medium' and loaded_state and 'encoder.0.weight' in loaded_state:
            first_layer_shape = loaded_state['encoder.0.weight'].shape
            if first_layer_shape[0] == 16:
                logger.info("Refined Detection: Flatten dim 4096 with 16 channels -> TINY model.")
                model_size_detected = 'tiny'
                
        return model_size_detected

    # Test Tiny
    detected_tiny = run_logic(tiny_state)
    print(f"Input: Tiny State (16ch). Detected: {detected_tiny}")
    if detected_tiny != 'tiny':
        print("FAIL: Expected 'tiny'")
        sys.exit(1)
        
    # Test Medium
    detected_medium = run_logic(medium_state)
    print(f"Input: Medium State (32ch). Detected: {detected_medium}")
    if detected_medium != 'medium':
        print("FAIL: Expected 'medium'")
        sys.exit(1)
        
    print("SUCCESS: Detection logic works correctly.")

if __name__ == "__main__":
    test_detection()
