
import asyncio
import logging
import torch
import numpy as np
from backend.modeling import VLABackbone

logger = logging.getLogger(__name__)

class AsyncPolicyServer:
    """
    Handles VLA inference requests asynchronously to ensure non-blocking operation
    for the robot control loop.
    """
    def __init__(self, model_name="openvla/openvla-7b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VLABackbone(model_name=model_name).to(self.device)
        self.model.eval()
        
        # Warmup
        logger.info("Warming up Policy Server...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.model.predict_action_chunk(dummy_input, "move forward")
    
    async def predict(self, image_bytes: bytes, instruction: str = "navigate"):
        """
        Async prediction endpoint.
        """
        # Decode image in thread to avoid blocking event loop
        img_tensor = await asyncio.to_thread(self._decode_image, image_bytes)
        img_tensor = img_tensor.to(self.device)
        
        # Run inference (computationally intensive, usually runs on GPU)
        # If running on CPU, might block, so to_thread is safer for the loop
        actions = await asyncio.to_thread(self.model.predict_action_chunk, img_tensor, instruction)
        
        return actions.cpu().numpy()

    def _decode_image(self, image_bytes):
        # Mock decode
        # In reality: cv2.imdecode -> transform -> tensor
        return torch.randn(1, 3, 224, 224) 
