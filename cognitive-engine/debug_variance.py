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
import json
import numpy as np
import sys
import os

# Add root to sys path
sys.path.append(r"c:\Users\chris\Embrained\software_suite")

from backend.models.latentslam import LatentSLAM
from backend.training.datasets.latentslam_dataset import LatentSLAMDataset

def debug_variance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"c:\Users\chris\Embrained\software_suite\data\tinyvae-latentslam_20260305_145007.pth"
    model = LatentSLAM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    # Random images
    with torch.no_grad():
        img = torch.rand(100, 3, 64, 64).to(device)
        z, a = torch.zeros(100, 128).to(device), torch.zeros(100, 2).to(device)
        mu, logvar = model.get_posterior(z, a, img)
        variances = torch.var(mu, dim=0)
        print(f"Random Images Max Variance of Mu (across dims): {variances.max().item():.6f}")
        print(f"Random Images Min Variance of Mu (across dims): {variances.min().item():.6f}")
        print(f"Random Images Mean Variance of Mu (across dims): {variances.mean().item():.6f}")
        print(f"Random Images Mean of Mu: {torch.mean(mu).item():.6f}")
        print(f"Random Images Mean of Exp(LogVar): {torch.exp(logvar).mean().item():.6f}")

if __name__ == '__main__':
    debug_variance()
