import os
import glob
import logging
import pickle
import numpy as np
import torch
from sklearn.decomposition import PCA
from PIL import Image
import threading
import time

# Use config if available, else defaults
try:
    from config import DATA_DIR, MODELS_DIR
except ImportError:
    DATA_DIR = "./data"
    MODELS_DIR = "./models"

logger = logging.getLogger("ManifoldService")

class ManifoldService:
    def __init__(self, vision_system=None):
        self.vision = vision_system
        self.pca = None
        self.manifold_points = [] # List of [x, y]
        self.is_ready = False
        self.lock = threading.Lock()
        
        # Determine paths
        self.cache_path = os.path.join(MODELS_DIR, "manifold.pkl")
        self.data_dir = DATA_DIR

    def start_background_fit(self):
        """Starts the PCA fitting process in a background thread."""
        t = threading.Thread(target=self._fit_process, daemon=True)
        t.start()
        
    def _fit_process(self):
        logger.info("Starting Manifold PCA process...")
        
        # 1. Check Cache
        if self._load_from_cache():
            return

        # 2. Collect Data
        if not self.vision:
            logger.warning("No VisionSystem provided. Cannot compute manifold.")
            return

        latents = self._collect_latents()
        
        if len(latents) < 50:
            logger.warning(f"Not enough data points ({len(latents)}) for robust PCA. Need at least 50.")
            # We can still fit if > n_components
            if len(latents) < 3:
                 logger.error("Insufficient data for PCA (need >= 3). Aborting.")
                 return

        # 3. Fit PCA
        try:
            logger.info(f"Fitting PCA on {len(latents)} points...")
            self.pca = PCA(n_components=2)
            transformed = self.pca.fit_transform(latents)
            
            # Normalize to -1..1 range for easier visualization? 
            # Or keep raw? T-SNE/PCA usually is arbitrary. 
            # Let's normalize min/max to fit in a box.
            
            # Save results
            with self.lock:
                self.manifold_points = transformed.tolist()
                self.is_ready = True
                
            self._save_to_cache()
            logger.info("Manifold PCA ready.")
            
        except Exception as e:
            logger.error(f"PCA Fit failed: {e}")

    def _collect_latents(self):
        """Scans data dir for images and runs them through the encoder."""
        latents = []
        
        # Scan all 'images' subdirectories
        # Matches: data/*/images/*.jpg
        pattern = os.path.join(self.data_dir, "*", "images", "*")
        files = glob.glob(pattern + ".jpg") + glob.glob(pattern + ".png")
        
        # Cap at some reasonable number to avoid taking forever? 
        # User said "previously collected data". Let's try 1000 random samples if too many?
        # Or just compute all. MobileNet is fast.
        
        if len(files) > 2000:
            logger.info(f"Found {len(files)} images. Subsampling 2000.")
            import random
            files = random.sample(files, 2000)
        else:
            logger.info(f"Found {len(files)} images.")

        count = 0
        self.vision.encoder.eval()
        
        with torch.no_grad():
            for f in files:
                try:
                    # We need the VisionSystem transform
                    # Accessing underlying transform might be hacky if not exposed.
                    # VisionSystem.transform is available.
                    
                    img = Image.open(f).convert('RGB')
                    # VisionSystem transform expects PIL or tensor? 
                    # vision.py: self.transform calls T.ToPILImage() first?
                    # Let's check vision.py again.
                    # T.Compose([T.ToPILImage(), ...]) means it expects Tensor or ndarray?
                    # Actually T.ToPILImage() handles ndarray or tensor. 
                    # If we pass PIL Image to ToPILImage, it might complain or pass through.
                    # Safer to convert to np array first.
                    
                    input_tensor = self.vision.transform(np.array(img)).unsqueeze(0).to(self.vision.device)
                    
                    # VisionSystem.encoder returns features
                    z = self.vision.encoder(input_tensor) # (1, 64)
                    latents.append(z.cpu().numpy().flatten())
                    
                    count += 1
                    if count % 100 == 0:
                        logger.info(f"Encoded {count}/{len(files)}")
                        
                except Exception as e:
                    pass
                    
        return np.array(latents)

    def _save_to_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'pca': self.pca,
                    'points': self.manifold_points
                }, f)
            logger.info(f"Saved manifold cache to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_from_cache(self):
        if not os.path.exists(self.cache_path):
            return False
            
        try:
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                
            with self.lock:
                self.pca = data['pca']
                self.manifold_points = data['points']
                self.is_ready = True
            
            logger.info("Loaded Manifold from cache.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def project(self, z):
        """Projects a single latent vector (64,) to 2D."""
        if not self.is_ready or self.pca is None:
            return None
        
        try:
            # PCA expects (n_samples, n_features)
            z_in = z.reshape(1, -1)
            res = self.pca.transform(z_in)
            return res[0].tolist() # [x, y]
        except Exception as e:
            return None
