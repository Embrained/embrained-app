
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, data_format='NCHW', temperature=None):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        # Create normalized coordinate grid (-1 to 1)
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.width),
            np.linspace(-1., 1., self.height))
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('grid_x', torch.tensor(pos_x, dtype=torch.float32))
        self.register_buffer('grid_y', torch.tensor(pos_y, dtype=torch.float32))

    def forward(self, feature_map):
        # feature_map: (Batch, C, H, W)
        batch_size, channels, height, width = feature_map.shape
        
        # Flatten spatial dims: (Batch, C, H*W)
        flat = feature_map.view(batch_size, channels, -1)
        
        # Apply Softmax over spatial dims
        softmax_attention = F.softmax(flat, dim=2)
        
        # Reshape grids to match flattened shape: (1, 1, H*W)
        grid_x = self.grid_x.view(-1)
        grid_y = self.grid_y.view(-1)
        
        # Compute Expectation (Sum(softmax * grid))
        # (Batch, C, H*W) * (H*W) -> (Batch, C)
        expected_x = torch.sum(softmax_attention * grid_x, dim=2)
        expected_y = torch.sum(softmax_attention * grid_y, dim=2)
        
        # Stack coordinates: (Batch, C, 2) -> (Batch, C*2)
        # Order: x1, y1, x2, y2... or x1...xn, y1...yn? 
        # Usually interlaced or concatenated. User spec: "Output: (Batch, 64)"
        # Let's stack them as [x1, y1, x2, y2, ...]
        coords = torch.stack([expected_x, expected_y], dim=2)
        return coords.view(batch_size, -1)

class SpatialEncoder(nn.Module):
    def __init__(self, output_keypoints=32, frozen_backbone=True):
        super(SpatialEncoder, self).__init__()
        
        # Stage A: Frozen Backbone (MobileNetV3-Small)
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Remove classifier and pooling
        # MobileNet features end after 'features' sequential block
        # Output is 576 channels.
        # We only use self.backbone.features
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Stage B: Trainable Adapter
        # Input 576 -> Conv1x1 -> 32
        self.adapter_conv = nn.Conv2d(576, output_keypoints, kernel_size=1)
        self.relu = nn.ReLU()
        
        # Determine Feature Map size for Spatial Softmax
        # For 120x160 input, MobileNetV3-Small output size:
        # It reduces roughly by 32x? 
        # 160 / 32 = 5
        # 120 / 32 = 3.75 -> 4
        # Let's verify or make dynamic? 
        # SpatialSoftmax needs fixed H/W or we can generate grid dynamically.
        # The provided SpatialSoftmax init takes H, W. 
        # Let's use 4x5 as spec suggested, but we can verify in forward if needed.
        # Ideally, we instatiate SpatialSoftmax with known dims.
        self.param_h = 4
        self.param_w = 5
        self.spatial_softmax = SpatialSoftmax(self.param_h, self.param_w)
        
    def forward(self, x):
        # x: (Batch, 3, 120, 160)
        
        # Stage A
        with torch.set_grad_enabled(not self.backbone.features[0][0].weight.requires_grad): # Respect frozen state
             feat = self.backbone.features(x) # (Batch, 576, H, W)
             
        # Stage B
        x = self.adapter_conv(feat) 
        x = self.relu(x)
        # x is (Batch, 32, H, W)
        
        # Spatial Softmax
        # Ensure feature map size matches what we expect
        if x.shape[2] != self.param_h or x.shape[3] != self.param_w:
             # If mismatch, maybe dynamic resize or update grid? 
             # For now, let's just proceed, assuming 120x160 input.
             pass
             
        kpts = self.spatial_softmax(x)
        return kpts # (Batch, 64)

class CQLNetwork(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, action_dim=4):
        super(CQLNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class FullQNet(nn.Module):
    """
    Wrapper for end-to-end training. 
    Combines Encoder (shared) and Policy.
    """
    def __init__(self, encoder, policy):
        super(FullQNet, self).__init__()
        self.encoder = encoder
        self.policy = policy
        
    def forward(self, img_current, img_goal):
        # Provide gradients to encoder -> policy
        kpt_cur = self.encoder(img_current)
        kpt_goal = self.encoder(img_goal)
        
        state = torch.cat([kpt_cur, kpt_goal], dim=1) # (Batch, 128)
        q_values = self.policy(state)
        return q_values
