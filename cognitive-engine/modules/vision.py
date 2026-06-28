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
import importlib.util
import logging
import cv2
import numpy as np
import numpy as np

from backend.utils import safe_import_torch
torch = safe_import_torch()

import torchvision.transforms as T
from config import MODELS_DIR, IMG_W, IMG_H, NORM_MEAN, NORM_STD
from config import MODELS_DIR, IMG_W, IMG_H, NORM_MEAN, NORM_STD
from modules.spatial_model import TinyVAE

class VisionSystem:
    def __init__(self, device='cpu', model_path=None):
        self.device = torch.device(device)
        self.groundtruth_mode = False
        self.extractor = None
        self.last_continuous_z = None  # Continuous embedding for discrete VQ-VAE manifold projection
            
        self.img_h = 64
        self.img_w = 64
        
        # Transform for TinyVAE
        # Resize only. No normalization for inference if model output is Sigmoid?
        # But we trained with [0,1] tensor. 
        # Transform for TinyVAE
        # Resize only. No normalization for inference if model output is Sigmoid?
        # But we trained with [0,1] tensor. 
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_h, self.img_w)),
            T.ToTensor()
        ])
        
        self.model_name = "TinyVAE"
        self.encoder = self._load_encoder(model_path)
        if self.encoder:
            self.encoder.to(self.device).eval()
        logging.debug(f"VisionSystem initialized on {self.device}")

    def load_model(self, path):
        """Loads weights from a specific path at runtime."""
        if not os.path.exists(path):
            logging.error(f"VisionSystem: Model path not found: {path}")
            return False
            
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            
            # Use helper to try matching sizes
            new_model = self._try_load_model_weights(state_dict, path)
            
            if new_model:
                self.encoder = new_model
                self.encoder.to(self.device).eval()
                self.model_name = os.path.basename(path)
                logging.debug(f"VisionSystem loaded new model: {self.model_name}")
                return True
            else:
                 return False
                 
        except Exception as e:
            logging.error(f"VisionSystem load failed: {e}")
            return False

    def _try_load_model_weights(self, state_dict, source_path_for_logging="Unknown"):
        """Tries to load state_dict into TinyVAE or Discrete VAE using robust size detection."""
        try:
            # 1. Detect size
            latent_dim, model_size, input_spatial_dim, in_channels = TinyVAE.detect_size(state_dict)
            logging.debug(f"Detected Base Encoder size: {model_size.upper()}, Latent: {latent_dim}, Spatial: {input_spatial_dim}, Channels: {in_channels} from {source_path_for_logging}")
            
            # 2. Instantiate natively via architecture detection
            vq_key = "vq.embedding.weight" if "vq.embedding.weight" in state_dict else "vq._embedding.weight" if "vq._embedding.weight" in state_dict else None
            if vq_key:
                from modules.spatial_model import DiscreteVQVAE
                num_embeddings = state_dict[vq_key].shape[0]
                model = DiscreteVQVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim, in_channels=in_channels, num_embeddings=num_embeddings)
            elif 'action_predictor.0.weight' in state_dict:
                # Contrastive Visuomotor Encoder (CVE)
                from modules.spatial_model import ContrastiveVisuomotorEncoder
                n_actions = state_dict['action_predictor.2.weight'].shape[0]  # output dim of action predictor
                model = ContrastiveVisuomotorEncoder(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim, in_channels=in_channels, n_actions=n_actions)
            elif 'fc_e.weight' in state_dict:
                from backend.models.quantized_spatial import DiscreteLatentSLAM
                try:
                    # Create discrete structure purely for encoder usage
                    model = DiscreteLatentSLAM(latent_dim=latent_dim, model_size=model_size, image_size=input_spatial_dim)
                except Exception as e:
                    logging.error(f"Failed to compile Discrete wrapper context: {e}")
                    return None
            else:
                model = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim, in_channels=in_channels)
            
            if hasattr(self, 'transform'):
                self.img_w = input_spatial_dim
                self.img_h = input_spatial_dim
                import torchvision.transforms as T
                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((self.img_h, self.img_w)),
                    T.ToTensor()
                ])
            
            # 3. Load
            try:
                model.load_state_dict(state_dict, strict=True)
                logging.debug("Loaded encoder weights with strict=True")
            except RuntimeError as e_strict:
                try:
                    model.load_state_dict(state_dict, strict=False)
                    logging.debug("Loaded encoder weights with strict=False")
                except Exception as e_loose:
                    # Attempt to load just the encoder if only encoder passed
                    new_sd = {}
                    for k, v in state_dict.items():
                        if k.startswith('encoder.') or not any(k.startswith(prefix) for prefix in ['decoder.', 'fc_mu.', 'fc_var.', 'fc_e.']):
                            new_sd[k.replace('encoder.', '')] = v
                            
                    if not new_sd or 'model_state_dict' in state_dict:
                        logging.debug("State dict appears to be a pure policy or lacks valid VAE keys. Skipping structural load.")
                        return None
                        
                    try:
                        model.encoder.load_state_dict(new_sd, strict=False)
                        logging.debug("Loaded localized encoder weights only")
                    except Exception as e_enc:
                        logging.error(f"Failed all weight loading attempts: {e_enc}")
                        return None
            
            return model
            
        except Exception as e:
            # We don't need to log this as an error if it's just rejecting a pure policy file
            if "fc_mu" in str(e) or "fc_e" in str(e):
                 logging.debug(f"File {source_path_for_logging} lacks internal VAE structure.")
            else:
                 logging.error(f"Could not load encoder model from {source_path_for_logging}: {e}")
            return None

    def _load_encoder(self, specific_path=None):
        """
        Loads TinyVAE and restores weights.
        """
        logging.debug("Loading TinyVAE Encoder...")
        
        candidates = []
        if specific_path:
             candidates.append(specific_path)
             
        # If specific path provided, try it.
        for path in candidates:
                 if os.path.exists(path):
                     try:
                         logging.debug(f"Found weights at {path}")
                         state_dict = torch.load(path, map_location=self.device, weights_only=True)
                         model = self._try_load_model_weights(state_dict, path)
                         if model:
                              return model
                     except Exception as e:
                         logging.warning(f"Could not load {path}: {e}")
        
        # Fallback: Don't load anything if no specific path
        logging.debug("Latent Space disabled (No VAE weights loaded).")
        return None

    def enable_groundtruth(self, enabled=True):
        """Toggle Ground Truth Mode ON permanently (bypasses neural vision)."""
        self.groundtruth_mode = enabled
        if enabled:
            from scripts.extract_telemetry import TelemetryExtractor
            self.extractor = TelemetryExtractor(datasets_dirs=[])
            
            # Look for telemetry_cache.npz anywhere reachable
            import glob
            from config import DATA_DIR
            search = glob.glob(os.path.join(DATA_DIR, '**', 'telemetry_cache.npz'), recursive=True)
            if search:
                success = self.extractor.load_cache(search[0])
                if success:
                     logging.info(f"Loaded TelemetryExtractor Cache from {search[0]}")
                else:
                     logging.error(f"Failed to load Telemetry Cache from {search[0]}")
            else:
                 logging.warning("No telemetry_cache.npz found! Single Frame GT Extraction will FAIL unless calibrated.")
                 
            self.encoder = None
            logging.info("Vision pipeline locked to Ground Truth Analytics. VAE encoder purged.")
            
    def process_frame(self, frame_input, webcam_input=None):
        """
        Decodes Bytes -> Tensor -> Keypoints (64-dim)
        OR
        Processes Numpy Image -> Tensor -> Keypoints (64-dim)
        [NEW] If dual-camera architecture (in_channels=6), stacks webcam_input below frame_input.
        """
        if frame_input is None:
            return None, None
            
        def _parse_img(raw_input):
            if isinstance(raw_input, np.ndarray):
                return raw_input
            elif raw_input is not None:
                nparr = np.frombuffer(raw_input, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return None

        # 1. Parse Internal Camera
        img = _parse_img(frame_input)
        if img is None:
            return None, None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        
        # 2. Parse External Camera (Allocentric-Only VAE Inference)
        webcam_img = _parse_img(webcam_input)
        if webcam_img is not None:
            if self.groundtruth_mode and self.extractor:
                 # Route dynamically through Ground Truth extraction
                 webcam_gray = cv2.cvtColor(webcam_img, cv2.COLOR_BGR2GRAY)
                 telemetry = self.extractor.process_single_frame(webcam_gray)
                 if telemetry:
                     # Flatten Dict into 4D PyTorch native Float Tensor format (matching CQL DataLoader)
                     mu_out = np.array([
                         telemetry['cx_norm'], 
                         telemetry['cy_norm'], 
                         telemetry['cos_yaw'], 
                         telemetry['sin_yaw']
                     ], dtype=np.float32)
                     return img, mu_out
                 else:
                     # Null tracking Frame loss
                     return img, np.zeros(4, dtype=np.float32)

            # We no longer override the tensor with the webcam image for VAE inference.
            # Only use internal images to estimate locations in latent space.
            pass
        elif self.groundtruth_mode:
             # Prevent crashes if webcam not plugged in but GT mode requested
             return img, np.zeros(4, dtype=np.float32)

        tensor = tensor.unsqueeze(0).to(self.device) # Add batch dim
        
        # Inference (Only if encoder exists)
        mu_out = None
        if self.encoder is not None:
            # Safe call
            with torch.no_grad():
                # Discrete encoders return (quantized, loss, perp, idx) while continuous return (recon, mu, var)
                output = self.encoder(tensor) 
                
                # Dynamic architecture inference
                if 'Contrastive' in type(self.encoder).__name__:
                    z = self.encoder.encode(tensor)
                    self.last_continuous_z = z.cpu().numpy().flatten()
                    mu = z  # CVE outputs continuous 32-dim directly
                elif 'Discrete' in type(self.encoder).__name__:
                    x_enc = self.encoder.encoder(tensor)
                    z_e = self.encoder.fc_e(x_enc)
                    
                    # Store continuous embedding for manifold projection (PCA expects this dimensionality)
                    self.last_continuous_z = z_e.cpu().numpy().flatten()
                    
                    # Handle both VQ naming conventions (embedding vs _embedding)
                    vq = self.encoder.vq
                    codebook = vq.embedding.weight if hasattr(vq, 'embedding') else vq._embedding.weight
                    d = torch.sum(z_e ** 2, dim=-1, keepdim=True) + \
                        torch.sum(codebook ** 2, dim=1) - \
                        2 * torch.matmul(z_e, codebook.t())
                        
                    idx = torch.argmin(d, dim=-1)
                    import torch.nn.functional as F
                    num_codes = vq.num_embeddings if hasattr(vq, 'num_embeddings') else vq._num_embeddings
                    mu = F.one_hot(idx, num_classes=num_codes).float()
                else: 
                    # Continuous returns: recon, mu, var
                    mu = output[1]
                    self.last_continuous_z = mu.cpu().numpy().flatten()
                
            if mu is not None:
                mu_out = mu.cpu().numpy().flatten()
            
        return img, mu_out
