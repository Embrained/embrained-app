import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pickle
import logging

# Configure Logging
logger = logging.getLogger("GetEmbeddings")
logging.basicConfig(level=logging.INFO)

class EmbeddingGenerator:
    def __init__(self, data_root):
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self):
        logger.info(f"Loading MobileNetV3-Small on {self.device}...")
        try:
            # Load pretrained MobileNetV3-Small
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            # Remove classification layer to get features (576 dim)
            model.classifier = nn.Identity()
            model.to(self.device).eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def generate(self, episodes_file="episodes.json", output_file="embeddings.pkl"):
        episodes_path = os.path.join(self.data_root, episodes_file)
        if not os.path.exists(episodes_path):
            raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

        logger.info("Loading episodes...")
        with open(episodes_path, 'r') as f:
            episodes = json.load(f)

        # Collect unique image paths
        unique_paths = set()
        for ep in episodes:
            if 'start_frame' in ep and 'image_path' in ep['start_frame']:
                unique_paths.add(ep['start_frame']['image_path'])
            if 'goal_frame' in ep and 'image_path' in ep['goal_frame']:
                unique_paths.add(ep['goal_frame']['image_path'])
            if 'actions' in ep:
                for action in ep['actions']:
                    if 'image_path' in action:
                        unique_paths.add(action['image_path'])

        logger.info(f"Found {len(unique_paths)} unique images to process.")
        
        embeddings = {}
        processed_count = 0
        
        with torch.no_grad():
            for rel_path in unique_paths:
                full_path = os.path.join(self.data_root, rel_path)
                
                if not os.path.exists(full_path):
                    logger.warning(f"Image not found: {full_path}")
                    continue
                
                try:
                    img = Image.open(full_path).convert('RGB')
                    input_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    # raw_z shape: [1, 576]
                    raw_z = self.model(input_tensor)
                    
                    # Normalization: z_norm = z / ||z||_2
                    # standard euclidean norm over latent dimension 1
                    norm_z = torch.nn.functional.normalize(raw_z, p=2, dim=1)
                    
                    # Convert to numpy and flatten
                    vec = norm_z.cpu().numpy().flatten() # (576,)
                    
                    embeddings[rel_path] = vec
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count}/{len(unique_paths)}")
                        
                except Exception as e:
                    logger.error(f"Error processing {rel_path}: {e}")

        output_path = os.path.join(self.data_root, output_file)
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test execution
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    gen = EmbeddingGenerator(data_dir)
    gen.generate()
