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
import numpy as np
import argparse
import sys

# Add parent directory to path to allow imports from backend/modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.vision import VisionSystem
from config import MODELS_DIR, DATA_DIR

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Encode raw goal images to goals.npy")
    parser.add_argument('--input', type=str, default=os.path.join(DATA_DIR, 'goals'), 
                        help="Directory containing goal images (jpg, png)")
    parser.add_argument('--output', type=str, default=os.path.join(MODELS_DIR, 'goals.npy'),
                        help="Path to save the output numpy array")
    parser.add_argument('--model', type=str, default=None,
                        help="Optional specific VAE model file path to use instead of the default")
    args = parser.parse_args()

    # 1. Setup
    if not os.path.exists(args.input):
        logging.error(f"Input directory does not exist: {args.input}")
        return

    image_files = sorted(glob.glob(os.path.join(args.input, "*.jpg")) + 
                         glob.glob(os.path.join(args.input, "*.png")))
    
    if not image_files:
        logging.error(f"No images found in {args.input}")
        return

    logging.info(f"Found {len(image_files)} images. Initializing Vision System...")

    # 2. Init Vision (Loads VAE)
    try:
        # device='cpu' is safer for utility scripts unless large batch, 
        # but let's use cuda if available for speed.
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vision = VisionSystem(device=device, model_path=args.model)
    except Exception as e:
        logging.error(f"Failed to initialize VisionSystem: {e}")
        return

    latents = []

    # 3. Process
    for img_path in image_files:
        logging.info(f"Processing {os.path.basename(img_path)}...")
        try:
            with open(img_path, 'rb') as f:
                file_bytes = f.read()
            
            # process_frame handles decoding, resizing, normalizing, and inference
            _, z = vision.process_frame(file_bytes)
            
            if z is not None:
                latents.append(z)
            else:
                logging.warning(f"Failed to encode {img_path}")
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")

    # 4. Save
    if latents:
        goals_arr = np.array(latents, dtype=np.float32)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        np.save(args.output, goals_arr)
        logging.info(f"Successfully saved {len(latents)} goal vectors to {args.output}")
        logging.info(f"Shape: {goals_arr.shape}")
    else:
        logging.warning("No latents generated. Nothing saved.")

if __name__ == "__main__":
    main()
