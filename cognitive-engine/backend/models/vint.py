import torch
import torch.nn as nn
import torchvision.models as models

class ViNTModel(nn.Module):
    def __init__(self, context_size=3, num_actions=4, freeze_backbone=True):
        super().__init__()
        self.context_size = context_size
        
        # 1. Vision Backbone
        # Use efficientnet_b0, replacing the final classifier
        eff_net = models.efficientnet_b0(weights='DEFAULT')
        self.encoder_dim = eff_net.classifier[1].in_features  # usually 1280
        eff_net.classifier = nn.Identity()
        self.vision_encoder = eff_net
        
        self.train_backbone = not freeze_backbone
        
        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
        # We need self.context_size history tokens + 1 goal token
        self.seq_len = context_size + 1
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.encoder_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 3. Action Head (Discrete)
        self.action_head = nn.Sequential(
            nn.Linear(self.encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_actions)
        )

    def train(self, mode=True):
        super().train(mode)
        if not self.train_backbone:
            self.vision_encoder.eval()

    def forward(self, obs_hist, goal):
        """
        obs_hist: (B, H, C, H_img, W_img) where H is context_size
        goal: (B, C, H_img, W_img)
        """
        B, H_seq, C, H_img, W_img = obs_hist.shape
        assert H_seq == self.context_size, f"Expected context size {self.context_size}, got {H_seq}"
        
        # Encode history
        obs_flat = obs_hist.reshape(B * H_seq, C, H_img, W_img)
        
        # If backbone is frozen, we can explicitly avoid saving any intermediate activations
        if not getattr(self, 'train_backbone', False):
            with torch.no_grad():
                obs_features = self.vision_encoder(obs_flat)  # (B*H_seq, 1280)
                goal_features = self.vision_encoder(goal).unsqueeze(1)  # (B, 1, 1280)
        else:
            obs_features = self.vision_encoder(obs_flat)
            goal_features = self.vision_encoder(goal).unsqueeze(1)
            
        obs_features = obs_features.reshape(B, H_seq, self.encoder_dim)
        
        # Concat along sequence dimension
        # Sequence: [obs_tt-(H-1), ..., obs_t, goal]
        seq = torch.cat([obs_features, goal_features], dim=1)  # (B, H_seq+1, 1280)
        
        # Add positional embedding
        seq = seq + self.pos_embedding
        
        # Pass through transformer
        # (B, S, E) -> (B, S, E)
        out_seq = self.transformer(seq)
        
        # Take the output corresponding to the goal token
        goal_token_out = out_seq[:, -1, :]  # (B, 1280)
        
        # Predict discrete action logits
        action_logits = self.action_head(goal_token_out)  # (B, num_actions)
        
        return action_logits
