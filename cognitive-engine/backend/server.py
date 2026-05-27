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


import asyncio
import logging
from backend.utils import safe_import_torch
torch = safe_import_torch()
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
        logger.debug("Warming up Policy Server...")
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
