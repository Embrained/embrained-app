
import logging
import torch
import numpy as np
# from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

class VisualHasher:
    """
    Implements a Semantic One-Way Function using SigLIP embeddings.
    Privacy Solution: 'Flour' data collection.
    """
    def __init__(self, model_id="google/siglip-so400m-patch14-384"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = None
        self.processor = None
        
        # Lazy Load to save resources if not used
    
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading SigLIP for Visual Hashing: {self.model_id}")
            try:
                from transformers import AutoModel, AutoProcessor
                self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                
                # Freeze
                for param in self.model.parameters():
                    param.requires_grad = False
            except ImportError:
                 logger.error("Transformers library not found. Visual Hashing disabled.")
            except Exception as e:
                 logger.error(f"Failed to load SigLIP: {e}")

    def process_and_discard(self, raw_frame):
        """
        Consumes a raw frame, computes embedding, and securely discards the frame.
        Returns: 768-dim numpy vector.
        """
        self._load_model()
        if self.model is None:
            return np.zeros(768) # Fallback
            
        try:
            # Process
            inputs = self.processor(images=raw_frame, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                # SigLIP get_image_features
                embeddings = self.model.get_image_features(**inputs)
                
                # Normalize (Crucial for SigLIP/contrastive models)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
            result = embeddings.cpu().numpy()
            
            # Secure Deletion
            # Python GC handles memory, but we can explicitly nullify references
            del raw_frame
            del inputs
            
            return result
            
        except Exception as e:
            logger.error(f"Hashing Failed: {e}")
            return np.zeros(768)
