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
import torch.nn as nn
import torch.nn.functional as F

class LatentSLAM(nn.Module):
    def __init__(self, latent_dim=32, num_actions=3, hidden_dim=256, image_size=64, model_size='large'):
        super(LatentSLAM, self).__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.image_size = image_size
        self.model_size = model_size.lower()
        
        if self.model_size == 'small':
            self.base_channels = 32
            self.n_layers = 4
        elif self.model_size == 'medium':
            self.base_channels = 64
            self.n_layers = 4
        elif self.model_size == 'enormous':
            self.base_channels = 128
            self.n_layers = 5
        elif self.model_size == 'tectonic':
            self.base_channels = 128
            self.n_layers = 6
        else: # Large
            self.base_channels = 64
            self.n_layers = 5

        # --- 1. SPATIAL ENCODER (Standard VAE) ---
        modules = []
        in_channels = 3
        current_channels = self.base_channels
        
        # 1. Initial Conv
        modules.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, stride=1, padding=1))
        modules.append(nn.ReLU())
        
        # 2. Downsampling Stack
        for i in range(self.n_layers):
            out_channels = min(current_channels * 2, 512)
            modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            modules.append(nn.ReLU())
            current_channels = out_channels
            
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)
        
        final_spatial = image_size // (2 ** self.n_layers)
        self.flattened_size = current_channels * final_spatial * final_spatial
        self.final_channels = current_channels
        self.spatial_size = final_spatial
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # --- 2. SPATIAL DECODER (Standard VAE) ---
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        
        dec_modules = []
        dec_modules.append(nn.Unflatten(1, (self.final_channels, self.spatial_size, self.spatial_size)))
        
        for i in range(self.n_layers):
            is_last = (i == self.n_layers - 1)
            target_out = current_channels // 2 if not is_last else 3
            
            if not is_last and target_out < self.base_channels:
                    target_out = self.base_channels
            
            if is_last:
                dec_modules.append(nn.ConvTranspose2d(current_channels, 3, kernel_size=4, stride=2, padding=1))
                dec_modules.append(nn.Sigmoid())
            else:
                dec_modules.append(nn.ConvTranspose2d(current_channels, target_out, kernel_size=4, stride=2, padding=1))
                dec_modules.append(nn.ReLU())
                current_channels = target_out
                
        self.decoder = nn.Sequential(*dec_modules)
        
        # --- 3. TRANSITION MLP (Discrete Odometry Prediction) ---
        # Predicts s_{t+1} given s_t for ALL actions
        self.transition_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_actions)
        )

    def encode(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x = self.decoder_input(z)
        return self.decoder(x)
        
    def predict_next_state(self, state):
        """Discrete forward prediction for latency across all actions."""
        out = self.transition_model(state)
        # Reshape to [batch, num_actions, latent_dim]
        return out.view(-1, self.num_actions, self.latent_dim)

    def forward(self, curr_image, action=None):
        """
        Standard VAE forward pass. 
        Note: The transition prediction happens separately in the training loop
        by calling `.predict_next_state()` to decouple the gradients explicitly.
        """
        mu, logvar = self.encode(curr_image)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
