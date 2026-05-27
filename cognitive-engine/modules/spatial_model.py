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
import math

import torch.nn as nn
import torch.nn.functional as F

class TinyVAE(nn.Module):
    """
    Standard VAE for 64x64 input.
    """
    @staticmethod
    def detect_size(state_dict):
        """Introspects checkpoint weights to determine model config."""
        
        # 1. Latent Dim
        if 'fc_mu.weight' in state_dict:
            fc_weight_shape = state_dict['fc_mu.weight'].shape
        elif 'fc_e.weight' in state_dict:
            fc_weight_shape = state_dict['fc_e.weight'].shape
        else:
            raise KeyError("Neither 'fc_mu.weight' nor 'fc_e.weight' found in state_dict")
            
        latent_dim = fc_weight_shape[0]
        flatten_dim = fc_weight_shape[1]
        
        # 2. Base Channels & In_Channels
        conv1_shape = state_dict['encoder.0.weight'].shape
        base_channels = conv1_shape[0]
        in_channels = conv1_shape[1]

        model_size = 'large' # Default fallback
        input_spatial_dim = 64
        
        # Detect number of layers
        conv_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('encoder.') and k.endswith('.weight')]
        if conv_indices:
            n_layers = len(conv_indices) - 1
        else:
            n_layers = 5 # Default

        if base_channels == 32 and n_layers == 4:
            model_size = 'small'
        elif base_channels == 64 and n_layers == 4:
            model_size = 'medium'
        elif base_channels == 64 and n_layers == 5:
            model_size = 'large'
        elif base_channels == 128 and n_layers == 5:
            model_size = 'enormous'
        elif base_channels == 128 and n_layers == 6:
            model_size = 'tectonic'
        else:
            # Fallback based on flatten_dim if base_channels/n_layers don't match
            if flatten_dim == 8192:
                model_size = 'medium' if base_channels == 64 else 'small'
            elif flatten_dim == 2048:
                model_size = 'enormous' if base_channels == 128 else 'large'
            elif flatten_dim == 512:
                model_size = 'tectonic'
            elif flatten_dim == 4096:
                model_size = 'tiny'
                
        # Infer current_channels at the end of encoder
        current_channels = base_channels
        for _ in range(n_layers):
            current_channels = min(current_channels * 2, 512)

        # Approximate spatial dim based on channels
        spatial = 1
        # This loop finds the smallest power of 2 such that spatial*spatial*current_channels >= flatten_dim
        # It assumes the final feature map is square.
        while (spatial * spatial * current_channels) < flatten_dim:
            spatial *= 2
        
        # The input spatial dim is the final spatial dim multiplied by 2 for each downsampling layer
        input_spatial_dim = spatial * (2 ** n_layers)
            
        return latent_dim, model_size, input_spatial_dim, in_channels

    def __init__(self, latent_dim=32, model_size='large', input_spatial_dim=64, in_channels=3):
        super(TinyVAE, self).__init__()
        
        self.model_size = model_size.lower()
        
        # Defaults based on size
        if self.model_size == 'tiny':
            self.base_channels = 16
            self.n_layers = 3
        elif self.model_size == 'small':
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
        else: # Large (Default)
            self.base_channels = 64
            self.n_layers = 5 # Deeper
            
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        modules = []
        current_channels = self.base_channels
        
        # 1. Initial Conv
        modules.append(nn.Conv2d(self.in_channels, current_channels, kernel_size=3, stride=1, padding=1))
        modules.append(nn.ReLU())
        
        # 2. Downsampling Stack
        for i in range(self.n_layers):
            out_channels = min(current_channels * 2, 512)
            modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            modules.append(nn.ReLU())
            current_channels = out_channels
            
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)
        
        # Calculate Flatten Dim
        # 64 -> 32 -> 16 -> 8 -> 4 (4 layers) -> 4x4 * channels
        # If 5 layers: -> 2 (2x2 * channels)
        final_spatial = input_spatial_dim // (2 ** self.n_layers)
        self.flatten_dim = current_channels * final_spatial * final_spatial
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        dec_modules = []
        dec_modules.append(nn.Unflatten(1, (current_channels, final_spatial, final_spatial)))
        
        for i in range(self.n_layers):
            # Going up
            # If last layer, output 3
            # Else output half channels
            
            # Heuristic for base channels constraint
            is_last = (i == self.n_layers - 1)
            target_out = current_channels // 2 if not is_last else 3
            
            if not is_last and target_out < self.base_channels:
                    target_out = self.base_channels
            
            if is_last:
                dec_modules.append(nn.ConvTranspose2d(current_channels, self.in_channels, kernel_size=4, stride=2, padding=1))
                dec_modules.append(nn.Sigmoid())
            else:
                dec_modules.append(nn.ConvTranspose2d(current_channels, target_out, kernel_size=4, stride=2, padding=1))
                dec_modules.append(nn.ReLU())
                current_channels = target_out
                
        self.decoder = nn.Sequential(*dec_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(self.decoder_input(z))
        return recon, mu, logvar

class VectorQuantizer(nn.Module):
    """
    EMA-based Vector Quantizer with dead code restart.
    commitment_cost=0.25 (standard VQ-VAE) prevents encoder collapse.
    decay=0.99 (standard EMA) stabilizes codebook vectors.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5, restart_threshold=0.5):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.restart_threshold = restart_threshold

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_(0, 1)

        # EMA tracking buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embedding_sum', self.embedding.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(num_embeddings))

    def forward(self, inputs):
        # inputs: [B, D]
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        if self.training:
            batch_cluster_size = encodings.sum(0)
            batch_embedding_sum = encodings.t() @ inputs.detach()

            self.ema_cluster_size.data.mul_(self.decay).add_(batch_cluster_size, alpha=1 - self.decay)
            self.ema_embedding_sum.data.mul_(self.decay).add_(batch_embedding_sum, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.epsilon)
                           / (n + self.num_embeddings * self.epsilon) * n)

            self.embedding.weight.data.copy_(self.ema_embedding_sum / cluster_size.unsqueeze(1))

            # DEAD CODE RESTART
            # Use ema_cluster_size to properly decay unused codes over time
            dead_mask = self.ema_cluster_size < self.restart_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and inputs.shape[0] > 0:
                random_indices = torch.randint(0, inputs.shape[0], (n_dead,), device=inputs.device)
                self.embedding.weight.data[dead_mask] = inputs[random_indices].detach() + torch.randn(n_dead, self.embedding_dim, device=inputs.device) * 0.01
                # Reset EMA buffers so the new code survives at least a few batches
                self.ema_cluster_size[dead_mask] = 1.0
                self.ema_embedding_sum[dead_mask] = self.embedding.weight.data[dead_mask].clone()

        loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())
        
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices

class DiscreteVQVAE(TinyVAE):
    """
    VQ-VAE inheriting from TinyVAE to keep the encoder/decoder stacks identical.
    Replaces the Gaussian sampling bottleneck with a discrete Vector Quantizer.
    """
    def __init__(self, latent_dim=32, model_size='large', input_spatial_dim=64, in_channels=3, num_embeddings=128):
        super(DiscreteVQVAE, self).__init__(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=input_spatial_dim, in_channels=in_channels)
        self.num_embeddings = num_embeddings
        
        # Disable unused continuous fc layers from parent
        self.fc_mu = None
        self.fc_logvar = None
        
        # Replace continuous fc_mu and fc_logvar with a single projection to embedding_dim
        self.fc_e = nn.Linear(self.flatten_dim, latent_dim)
        
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        
    def forward(self, x):
        x_enc = self.encoder(x)
        z_e = self.fc_e(x_enc)
        
        quantized, vq_loss, perplexity, encodings = self.vq(z_e)
        
        recon = self.decoder(self.decoder_input(quantized))
        return recon, quantized, vq_loss, perplexity


class OracleQNetwork(nn.Module):
    def __init__(self, state_dim=4, goal_dim=4, action_dim=5, **kwargs):
        super().__init__()
        input_dim = state_dim + goal_dim
        # Keep input layer for shape introspection
        self.input_layer = nn.Linear(input_dim, 256)
        
        self.net = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Compatibility attribute for Planner logic
        self.use_ln = False
        
    def forward(self, state, goal=None):
        if goal is not None:
             x = torch.cat([state, goal], dim=-1)
        else:
             x = state
        return self.net(x)

class FixedGoalBCNetwork(nn.Module):
    def __init__(self, state_dim=32, action_dim=4, **kwargs):
        super().__init__()
        # Input: 3 temporally stacked latents
        input_dim = state_dim * 3
        
        self.input_layer = nn.Linear(input_dim, 256)
        
        self.net = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.use_ln = False
        
    def forward(self, state, goal=None):
        return self.net(state)

class IRPredictorNetwork(nn.Module):
    def __init__(self, state_dim=96, action_dim=3, **kwargs):
        super().__init__()
        # State dim + action (one-hot or embedding, we'll just pass action as an integer and embed it)
        # Actually since Planner passes `obs` which is just stacked states, and tests actions,
        # we can just take [state, action_id] where action_id is embedded
        self.action_embed = nn.Embedding(5, 8) # actions F, L, R => indices, up to 5
        input_dim = state_dim + 8
        
        # input layer for introspection matching
        self.input_layer = nn.Linear(input_dim, 128)
        
        self.net = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Predicts scalar IR reading
        )
        self.use_ln = False
        
    def forward(self, state, action_idx):
        if action_idx.dim() == 0:
            action_idx = action_idx.unsqueeze(0)
        a_emb = self.action_embed(action_idx)
        # Check if state has batch dim
        if state.dim() == 1:
            state = state.unsqueeze(0)
            a_emb = a_emb.unsqueeze(0)
        if a_emb.dim() == 1:
            a_emb = a_emb.unsqueeze(0)
            
        x = torch.cat([state, a_emb], dim=-1)
        return self.net(x)

class CQLNetwork(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=256, action_dim=5, use_ln=True, model_size='large'):
        super(CQLNetwork, self).__init__()
        self.use_ln = use_ln
        self.model_size = model_size.lower()
        
        # Simple Logic
        if self.model_size == 'small':
            hidden_dim = 256
            layers = 2
        elif self.model_size == 'medium':
            hidden_dim = 512
            layers = 3
        elif self.model_size == 'enormous':
            hidden_dim = 2048
            layers = 5
        elif self.model_size == 'tectonic':
            hidden_dim = 4096
            layers = 6
        else: # Large
            hidden_dim = 1024
            layers = 4
            
        if self.use_ln:
            # We only apply LayerNorm to the latent parts (Current + Goal). 
            # If the architecture has cleanly dropped the 2 explicit state dims, normalize the entire vector!
            if input_dim in [12, 16]:
                # Pure Oracle Configuration! No LayerNorm should ever be applied to raw geometric telemetry!
                self.use_ln = False
            else:
                latent_dim_total = input_dim if input_dim % 32 == 0 else input_dim - 2
                self.ln = nn.LayerNorm(latent_dim_total)
                self.latent_dim_total = latent_dim_total
            
            
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        if self.use_ln and hasattr(self, 'latent_dim_total'):
            # Split the input: LayerNorm only the latent portion
            latent_part = x[:, :self.latent_dim_total]
            state_part = x[:, self.latent_dim_total:]
            
            latent_norm = self.ln(latent_part)
            x = torch.cat([latent_norm, state_part], dim=1)
        elif self.use_ln:
            x = self.ln(x)
            
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

class FullQNet(nn.Module):
    """
    Wrapper for end-to-end training (or pseudo-end-to-end where encoder is frozen). 
    """
    def __init__(self, encoder, policy, has_goal=True):
        super(FullQNet, self).__init__()
        self.encoder = encoder
        self.policy = policy
        self.has_goal = has_goal
        
    def forward(self, img_current, img_goal, state_cur):
        # img_current: [B, 4, 3, H, W]
        # img_goal: [B, 3, H, W]
        # state_cur: [B, 3]
        
        B = img_current.size(0)
        
        # Reshape stacked images to process all frames simultaneously: [B*4, 3, H, W]
        img_current_flat = img_current.view(-1, img_current.size(-3), img_current.size(-2), img_current.size(-1))
        
        feat_cur = self.encoder.encoder(img_current_flat)
        mu_cur_flat = self.encoder.fc_mu(feat_cur) # [B*4, 32]
        
        # Reshape back to [B, 4, 32] and then flatten time dim to [B, 128]
        mu_cur = mu_cur_flat.view(B, -1)
        
        # Single goal image processed normally
        if self.has_goal and img_goal is not None:
            feat_goal = self.encoder.encoder(img_goal)
            mu_goal = self.encoder.fc_mu(feat_goal) # [B, 32]
            
            # Concatenate purely spatial dimensions [Latent Current : 32] + [Latent Goal : 32] = 64
            # Drops the state_cur variables completely from the matrix arrays!
            state = torch.cat([mu_cur, mu_goal], dim=1) 
        else:
            state = mu_cur
            
        return self.policy(state)

class ValueNetwork(nn.Module):
    """
    Multilayer Perceptron that predicts the topological step distance between two latent states.
    Input: [z_current, z_goal]
    Output: Scalar (predicted steps)
    """
    def __init__(self, latent_dim=32, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        input_dim = latent_dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, z_cur, z_goal):
        # z_cur, z_goal: [B, latent_dim]
        if z_cur.dim() == 1:
            z_cur = z_cur.unsqueeze(0)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0)
            
        x = torch.cat([z_cur, z_goal], dim=-1)
        return self.net(x)
