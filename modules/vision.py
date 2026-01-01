
import os
import glob
import sys
import importlib.util
import logging
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from config import MODELS_DIR, IMG_W, IMG_H, NORM_MEAN, NORM_STD
from modules.spatial_model import SpatialEncoder

class VisionSystem:
    def __init__(self, device='cpu', model_path=None):
        self.device = torch.device(device)
        self.img_h = 120
        self.img_w = 160
        
        # New Transform for Spatial Softmax (matches training)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_h, self.img_w)), # 120x160
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model_name = "SpatialEncoder"
        self.encoder = self._load_encoder(model_path)
        self.encoder.to(self.device).eval()
        logging.info(f"VisionSystem initialized on {self.device}")

    def _load_encoder(self, specific_path=None):
        """
        Loads SpatialEncoder and restores weights.
        """
        logging.info("Loading SpatialEncoder...")
        try:
             # Initialize model structure
             model = SpatialEncoder(output_keypoints=32, frozen_backbone=True)
             
             candidates = []
             if specific_path:
                 candidates.append(specific_path)
                 
             # Fallback logic: check typical locations
             candidates.extend([
                 os.path.join(MODELS_DIR, "spatial_encoder.pth"),
                 os.path.join(os.path.dirname(MODELS_DIR), "data", "spatial_encoder.pth"),
                 os.path.join(MODELS_DIR, "..", "data", "spatial_encoder.pth")
             ])
             
             weights_loaded = False
             for path in candidates:
                 if os.path.exists(path):
                     logging.info(f"Found weights at {path}")
                     # Handling keys: Wrapper might have different keys? 
                     # train_cql saves q_net.encoder.state_dict(), so keys should match SpatialEncoder directly.
                     state_dict = torch.load(path, map_location=self.device)
                     model.load_state_dict(state_dict)
                     weights_loaded = True
                     break
            
             if not weights_loaded:
                 logging.warning("No spatial_encoder.pth found! Using random weights (Backbone is ImageNet).")
                 
             return model
        except Exception as e:
            logging.error(f"Failed to load SpatialEncoder: {e}")
            raise e

    def process_frame(self, frame_input):
        """
        Decodes Bytes -> Tensor -> Keypoints (64-dim)
        OR
        Processes Numpy Image -> Tensor -> Keypoints (64-dim)
        """
        if frame_input is None:
            return None, None
            
        # Decode / Handle Input
        img = None
        if isinstance(frame_input, np.ndarray):
             img = frame_input
        else:
            # Assume bytes
            nparr = np.frombuffer(frame_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, None
            
        # Preprocess
        # OpenCV is BGR. 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device) # Add batch dim
        
        # Inference
        with torch.no_grad():
            kpts = self.encoder(tensor) # (1, 64)
            
        return img, kpts.cpu().numpy().flatten()
