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

def random_crop(x, size=64, pad=4):
    """
    Applies Random Crop augmentation to a batch of images.
    x has shape (B, C, H, W)
    """
    n, c, h, w = x.shape
    
    # Pad images (using replicate reflection)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    
    # Generate random crop coordinates
    crop_w = torch.randint(0, pad * 2 + 1, (n,))
    crop_h = torch.randint(0, pad * 2 + 1, (n,))
    
    # Crop the padded images
    cropped = torch.zeros((n, c, size, size), device=x.device, dtype=x.dtype)
    for i in range(n):
        cropped[i] = x_padded[i, :, crop_h[i]:crop_h[i]+size, crop_w[i]:crop_w[i]+size]
        
    return cropped

def random_shift(x, pad=4):
    """
    Random shift commonly used in DrQ.
    Applies random padding and cropping to shift the image.
    """
    return random_crop(x, size=x.shape[-1], pad=pad)
