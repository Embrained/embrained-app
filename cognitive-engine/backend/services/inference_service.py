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

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os
from ..models.latentslam import LatentSLAM

class LatentSLAMInference:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from modules.spatial_model import TinyVAE

        # Dynamically detect architectural parameters from state_dict
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        try:
             latent_dim, model_size, image_size, _ = TinyVAE.detect_size(state_dict)
        except:
             latent_dim, model_size, image_size = 32, 'large', 64
             
        hidden_dim = 256
        model_size_lower = model_size.lower()
        if model_size_lower == "medium": hidden_dim = 512
        elif model_size_lower == "large": hidden_dim = 1024
        elif model_size_lower == "enormous": hidden_dim = 2048
        elif model_size_lower == "tectonic": hidden_dim = 4096
            
        self.model = LatentSLAM(
            latent_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            image_size=image_size, 
            model_size=model_size
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        self.current_latent = torch.zeros(1, self.model.latent_dim).to(self.device)

    def get_latent_state(self, image_bgr, last_action_pwm):
        """
        Estimates the current latent state.
        Args:
            image_bgr: Raw OpenCV frame (BGR)
            last_action_pwm: [left, right] PWM integers
        Returns:
            latent_mu (numpy array)
        """
        # Prepare Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Prepare Action
        action_tensor = torch.tensor([last_action_pwm], dtype=torch.float32).to(self.device) / 255.0
        
        with torch.no_grad():
            mu, log_var = self.model.encode(img_tensor)
            self.current_latent = mu # Update for next step
            
        return mu.cpu().numpy()[0]

    def encode_goal(self, image_bgr):
        """
        Encodes a goal image without updating the recurrent state.
        Uses a zero latent state prior.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.model.encode(img_tensor)
            
        return mu.cpu().numpy()[0]

    def predict_next_state(self, action_pwm):
        """
        Predicts where we will be based on motor command (Prior).
        """
        l, r = action_pwm[0], action_pwm[1]
        
        # [NEW] Stop Action Physics: Next state is identical to current state
        if abs(l) < 10 and abs(r) < 10:
             return self.current_latent.cpu().numpy()[0]
             
        action_idx = 0 # Forward
        if l < r: action_idx = 2    # right
        elif r < l: action_idx = 1  # left
        
        with torch.no_grad():
            hat_mu_all = self.model.predict_next_state(self.current_latent)
            # Pick the prediction for the given action
            mu = hat_mu_all[0, action_idx].unsqueeze(0)
            
        return mu.cpu().numpy()[0]
