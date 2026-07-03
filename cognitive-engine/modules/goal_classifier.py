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


from backend.utils import safe_import_torch
torch = safe_import_torch()

import torch.nn as nn
import torch.nn.functional as F
from modules.spatial_model import TinyVAE


class GoalClassifier(nn.Module):
    """Binary classifier for visual goal state detection.
    
    Reuses the TinyVAE encoder backbone (conv stack only, no VAE heads)
    to produce a binary classification: P(current frame is a goal state).
    
    Designed to be trained on user-curated positive examples (e.g., data/sofa/)
    vs. all other exploration frames as negatives.
    
    At inference time, the classifier can be used to:
      1. Provide continuous reward shaping for CQL training (P(goal) → reward)
      2. Detect terminal states (P(goal) > threshold → INTENTIONAL_STOP)
    """
    
    def __init__(self, latent_dim=32, model_size='large', input_spatial_dim=64, in_channels=3):
        super(GoalClassifier, self).__init__()
        
        # Reuse TinyVAE encoder backbone for architecture compatibility
        _helper = TinyVAE(
            latent_dim=latent_dim,
            model_size=model_size,
            input_spatial_dim=input_spatial_dim,
            in_channels=in_channels
        )
        self.encoder = _helper.encoder
        self.flatten_dim = _helper.flatten_dim
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Raw logit; sigmoid applied at inference
        )
    
    def forward(self, x):
        """Returns raw logit (use with BCEWithLogitsLoss for training)."""
        features = self.encoder(x)
        return self.head(features).squeeze(-1)
    
    def predict_proba(self, x):
        """Returns probability P(goal) ∈ [0, 1] for each image in the batch."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path, device='cpu'):
        """Load a trained GoalClassifier from a checkpoint file.
        
        Checkpoint format: {
            'model_state_dict': state_dict,
            'model_size': str,
            'latent_dim': int,
            'input_spatial_dim': int,
            'in_channels': int,
            'threshold': float,
            'metrics': dict,
        }
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = GoalClassifier(
                latent_dim=checkpoint.get('latent_dim', 32),
                model_size=checkpoint.get('model_size', 'large'),
                input_spatial_dim=checkpoint.get('input_spatial_dim', 64),
                in_channels=checkpoint.get('in_channels', 3),
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Bare state dict fallback
            model = GoalClassifier()
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        return model
