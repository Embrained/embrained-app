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
import torch.nn.functional as F
from backend.models.dreamerv3 import DreamerPolicy
import logging
import os

logger = logging.getLogger("DreamerController")

class DreamerController:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.state = 'INITIALIZING'
        self.last_latent = None
        self.h_t = None
        self.z_t = None
        self.prev_action = None
        
        logger.info(f"Loading DreamerV3 policy from {model_path} onto {device}...")
        
        try:
            state_dict_pkg = torch.load(model_path, map_location=device, weights_only=True)
            self.model_size = state_dict_pkg.get('model_size', 'small')
            self.latent_dim = state_dict_pkg.get('latent_dim', 32)
            
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from config import ACTION_DIM
            
            self.policy = DreamerPolicy(action_dim=ACTION_DIM, obs_dim=self.latent_dim, hidden_dim=256, state_dim=32).to(device)
            self.policy.load_state_dict(state_dict_pkg['model_state_dict'])
            self.policy.eval()
            self.action_dim = ACTION_DIM
            
            # Initialize RNN state
            self.reset_state()
            self.state = 'READY'
            logger.info("DreamerV3 Policy loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DreamerV3 Policy: {e}")
            self.state = 'ERROR'
            
    def reset_state(self):
        self.h_t = torch.zeros(1, self.policy.hidden_dim, device=self.device)
        self.z_t = torch.zeros(1, self.policy.state_dim * 2, device=self.device)
        self.prev_action = torch.zeros(1, self.action_dim, device=self.device)
        
    def get_action_from_latent(self, z_cur):
        """
        Takes the current VAE latent z_cur of shape (1, latent_dim).
        Returns integer action.
        """
        if self.state != 'READY' or z_cur is None:
            return 0 # STOP
            
        with torch.no_grad():
            z_curr = torch.FloatTensor(z_cur).to(self.device).view(1, -1)
                
            # Step World Model
            self.h_t, self.z_t, _, _ = self.policy.world_model.rssm.step(self.h_t, self.z_t, self.prev_action, z_curr)
            
            self.last_latent = self.z_t[0].cpu().numpy()
            
            # Get Action from Actor
            mean, std = self.policy.actor(self.h_t, self.z_t)
            action_idx = torch.argmax(mean, dim=-1).item()
            
            # Update prev action
            self.prev_action = F.one_hot(torch.tensor([action_idx], device=self.device), num_classes=self.action_dim).float()
            
            return action_idx

    def get_action(self, img=None, latent=None):
        """Standardized interface for engine."""
        if latent is not None:
             return self.get_action_from_latent(latent)
        return 0
