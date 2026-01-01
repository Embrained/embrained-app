
import os
import glob
import logging
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import torch
import cv2
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger("PCAService")

class PCAService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PCAService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.pca_model = None
        self.pca_path = os.path.join(os.path.dirname(__file__), "pca_model.pkl")
        self.global_latents = None
        
        # Hardcoded root for now, or pass it in? 
        # Ideally passed in, but singleton makes init matching hard.
        # We'll rely on generating it if missing.
        self.external_data_root = r"C:\Users\chris\ArtificialBrain\Explorer"
        
        self.initialized = True

    def initialize_vision(self):
        # Lazy import to avoid circular dep issues or early gpu init
        from modules.vision import VisionSystem
        if not hasattr(self, 'vision'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing VisionSystem on {device}...")
            self.vision = VisionSystem(device=device)

    def load_or_fit_pca(self, num_samples=10000):
        if self.pca_model is not None:
            return
            
        # Try loading from disk
        if os.path.exists(self.pca_path):
            try:
                with open(self.pca_path, 'rb') as f:
                    data = pickle.load(f)
                    self.pca_model = data['model']
                    self.global_latents = data.get('background_latents', None) # Optional to keep generic background
                logger.info("Loaded cached PCA model.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached PCA: {e}")

        # Fit new model
        logger.info("Fitting new PCA model...")
        
        # Borrow logic from visualize_pca loading
        pattern = os.path.join(self.external_data_root, "capture-*", "latents.npy")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError("No external latents found. Please run generate_capture_latents.py first.")
            
        all_latents = []
        for f in files:
            try:
                data = np.load(f)
                if data.ndim == 2 and data.shape[1] == 576:
                    all_latents.append(data)
            except:
                pass
                
        if not all_latents:
             raise ValueError("No valid latents found.")
             
        combined = np.concatenate(all_latents, axis=0)
        
        # Subsample for PCA fitting if huge
        if combined.shape[0] > num_samples:
             indices = np.random.choice(combined.shape[0], num_samples, replace=False)
             fitting_data = combined[indices]
             # Keep a subsample for background plotting too
             self.global_latents = fitting_data
        else:
            self.global_latents = combined
            
        self.pca_model = PCA(n_components=2)
        self.pca_model.fit(self.global_latents)
        
        # Cache it
        try:
            with open(self.pca_path, 'wb') as f:
                pickle.dump({'model': self.pca_model, 'background_latents': self.global_latents}, f)
        except Exception as e:
            logger.warning(f"Failed to cache PCA: {e}")

    def get_dataset_latents(self, dataset_path, progress_callback=None):
        """
        Get or generate latents for a specific local dataset (in ./data).
        """
        # Check for cached latents.npy
        latent_path = os.path.join(dataset_path, "latents.npy")
        if os.path.exists(latent_path):
            return np.load(latent_path)
            
        # Generate
        self.initialize_vision()
        
        img_dir = os.path.join(dataset_path, "images")
        if not os.path.exists(img_dir):
            # Fallback to root?
            img_dir = dataset_path
            
        jpgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if not jpgs:
            return np.array([])
            
        latents = []
        total = len(jpgs)
        
        for i, img_p in enumerate(jpgs):
            if i % 10 == 0 and progress_callback:
                progress_callback(f"Processing {i}/{total}...")
                
            img = cv2.imread(img_p)
            if img is None: continue
            
            _, lat = self.vision.process_frame(img)
            if lat is not None:
                latents.append(lat)
                
        if not latents:
            return np.array([])
            
        data = np.array(latents, dtype=np.float32)
        np.save(latent_path, data)
        return data

    def generate_plot(self, dataset_name, dataset_path, progress_cb=None):
        self.load_or_fit_pca()
        
        if progress_cb: progress_cb("Loading dataset latents...")
        ds_latents = self.get_dataset_latents(dataset_path, progress_cb)
        
        if len(ds_latents) == 0:
            raise ValueError("No images/latents in dataset.")
            
        if progress_cb: progress_cb("Generating Plot...")
        
        # Transform
        bg_2d = self.pca_model.transform(self.global_latents) if self.global_latents is not None else None
        ds_2d = self.pca_model.transform(ds_latents)
        
        # Plot
        plt.figure(figsize=(10, 6), dpi=100)
        
        # Background
        if bg_2d is not None:
            plt.scatter(bg_2d[:, 0], bg_2d[:, 1], c='lightgray', s=5, alpha=0.3, label='Global Manifold')
            
        # Dataset
        # Gradient colors to show time/sequence
        times = np.arange(len(ds_2d))
        
        # Connect dots with a thin black line
        plt.plot(ds_2d[:, 0], ds_2d[:, 1], c='black', linewidth=0.5, alpha=0.5, zorder=1)
        
        # Scatter with empty circles: map colors to edges
        norm = plt.Normalize(times.min(), times.max())
        colors = plt.cm.viridis(norm(times))
        
        # s=15 (slightly larger than 10 to make hollow visible), linewidths=0.5
        plt.scatter(ds_2d[:, 0], ds_2d[:, 1], facecolors='none', edgecolors=colors, s=15, linewidths=0.5, label=dataset_name, zorder=2)
        
        # Colorbar via ScalarMappable since we used manual colors
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Frame Index', ax=plt.gca())
        
        # Start/End markers
        plt.scatter(ds_2d[0, 0], ds_2d[0, 1], c='green', marker='^', s=100, label='Start', edgecolors='black')
        plt.scatter(ds_2d[-1, 0], ds_2d[-1, 1], c='red', marker='v', s=100, label='End', edgecolors='black')
        
        plt.title(f"Dataset Projection: {dataset_name}")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_str

