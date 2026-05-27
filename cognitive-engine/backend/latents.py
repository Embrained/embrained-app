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
import json
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import numpy as np
import logging

# Configure Logging
logger = logging.getLogger("LatentGenerator")

class LatentGenerator:
    def __init__(self, data_root):
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = T.Compose([
            T.Resize((224, 224)), # MobileNet standard input
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self):
        logger.info(f"Loading MobileNetV3-Small on {self.device}...")
        try:
            # Load pretrained MobileNetV3-Small
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            # Remove classification layer to get embeddings
            # The classifier is typically a Sequential; we want the output before the final classification
            # MobileNetV3 Small structure: features -> avgpool -> classifier
            # We can replace classifier with Identity to get the pooled features (576 dim)
            model.classifier = nn.Identity()
            model.to(self.device).eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def generate_latents_from_file(self, filename):
        """
        Reads a trajectory file (json), processes all images found, and saves latents.
        Returns the path to the saved latents file.
        """
        file_path = os.path.join(self.data_root, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing {filename}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Normalize data to a list of image paths
        image_paths = []
        
        # Handle different structures
        if isinstance(data, list):
            # Check first item
            if not data:
                return None
                
            first = data[0]
            if 'image_path' in first:
                # Flat list of transitions (all_transitions.json)
                image_paths = [item['image_path'] for item in data]
        
        # Deduplicate paths to avoid redundant processing?
        # Maybe, but order matters for sequence.
        # User said "save the resulting sequence". 
        # If we dedupe, we lose sequence.
        # But if the file is episodes, it's a collection of sequences.
        # If it's all_transitions, it's one long sequence (sorted by time).
        
        if not image_paths:
            raise ValueError("No image paths found in file.")

        latents = []
        valid_count = 0
        
        # Batch processing could be faster, but let's do simple loop first
        with torch.no_grad():
            for rel_path in image_paths:
                full_path = os.path.join(self.data_root, rel_path)
                
                # Check consistency if rel_path includes data root or not
                # engine.py loads relative to data_root? usually yes.
                
                try:
                    if not os.path.exists(full_path):
                        # Try relative to data_root parent?
                        # In training.py: img_rel_path = f"{ds_name}/images/{img_filename}"
                        # So full path is data_root/ds_name/images/...
                        logger.warning(f"Image not found: {full_path}")
                        latents.append(np.zeros(576, dtype=np.float32)) # Append zero or skip?
                        continue
                        
                    img = Image.open(full_path).convert('RGB')
                    input_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    feature = self.model(input_tensor) # [1, 576]
                    feature = feature.cpu().numpy().flatten()
                    
                    latents.append(feature)
                    valid_count += 1
                    
                    if valid_count % 100 == 0:
                        logger.info(f"Processed {valid_count}/{len(image_paths)}")
                        
                except Exception as e:
                    logger.error(f"Error processing {rel_path}: {e}")
                    latents.append(np.zeros(576, dtype=np.float32))

        # Save result
        output_filename = f"{os.path.splitext(filename)[0]}_latents.npy"
        output_path = os.path.join(self.data_root, output_filename)
        
        np.save(output_path, np.array(latents, dtype=np.float32))
        logger.info(f"Saved {len(latents)} latents to {output_filename}")
        
        return output_filename
