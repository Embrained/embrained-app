# Embrained - Neural Navigation Software Suite
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    EMA-based Vector Quantizer with dead code restart.
    Fixes codebook collapse by:
    1. N(0,1) initialization (matches encoder output scale)
    2. Exponential Moving Average codebook updates (no gradient needed for codebook)
    3. Dead code restart: unused entries are reinitialized from encoder outputs
    4. Commitment-only loss (EMA handles codebook updates)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=2.0,
                 decay=0.9, epsilon=1e-5, restart_threshold=1.0):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        self._restart_threshold = restart_threshold
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_(0, 1)

        # EMA tracking buffers (not saved as parameters — no gradients)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_embedding_sum', self._embedding.weight.data.clone())
        self.register_buffer('_usage_count', torch.zeros(num_embeddings))

    def forward(self, inputs):
        # Flatten inputs (if they aren't already)
        flat_inputs = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self._embedding.weight.t()))
                    
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        
        # EMA codebook updates (training only)
        if self.training:
            batch_cluster_size = encodings.sum(0)
            batch_embedding_sum = encodings.t() @ flat_inputs.detach()

            self._ema_cluster_size.data.mul_(self._decay).add_(batch_cluster_size, alpha=1 - self._decay)
            self._ema_embedding_sum.data.mul_(self._decay).add_(batch_embedding_sum, alpha=1 - self._decay)

            # Laplace smoothing to avoid division by zero
            n = self._ema_cluster_size.sum()
            cluster_size = ((self._ema_cluster_size + self._epsilon)
                           / (n + self._num_embeddings * self._epsilon) * n)

            self._embedding.weight.data.copy_(self._ema_embedding_sum / cluster_size.unsqueeze(1))

            # Track cumulative usage for dead code detection
            self._usage_count += batch_cluster_size
            
            # Restart dead codes by sampling from current batch encoder outputs
            dead_mask = self._usage_count < self._restart_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and flat_inputs.shape[0] > 0:
                random_indices = torch.randint(0, flat_inputs.shape[0], (n_dead,), device=inputs.device)
                self._embedding.weight.data[dead_mask] = flat_inputs[random_indices].detach() + torch.randn(n_dead, self._embedding_dim, device=inputs.device) * 0.01
                self._ema_cluster_size[dead_mask] = 1.0
                self._ema_embedding_sum[dead_mask] = self._embedding.weight.data[dead_mask].clone()
                self._usage_count[dead_mask] = 0

        # Commitment loss only (EMA handles codebook updates, no q_latent_loss needed)
        loss = self._commitment_cost * F.mse_loss(flat_inputs, quantized.view(-1, self._embedding_dim).detach())
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.reshape(inputs.shape[0])


class DiscreteLatentSLAM(nn.Module):
    def __init__(self, latent_dim=128, num_actions=3, hidden_dim=256, image_size=64, model_size='large', num_embeddings=128):
        super(DiscreteLatentSLAM, self).__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.image_size = image_size
        self.model_size = model_size.lower()
        self.num_embeddings = num_embeddings
        
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

        # --- 1. SPATIAL ENCODER ---
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
        
        self.fc_e = nn.Linear(self.flattened_size, latent_dim)
        
        # --- 2. VECTOR QUANTIZER ---
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        
        # --- 3. SPATIAL DECODER ---
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
        
        # --- 4. TRANSITION MLP (Categorical Prediction) ---
        # Predicts logits for the next codebook index given the current continuous quantized vector
        self.transition_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_embeddings * num_actions)
        )

    def encode(self, x):
        features = self.encoder(x)
        z_e = self.fc_e(features)
        z_q, vq_loss, perplexity, indices = self.vq(z_e)
        return z_e, z_q, vq_loss, perplexity, indices

    def decode(self, z_q):
        x = self.decoder_input(z_q)
        return self.decoder(x)
        
    def predict_next_state(self, state_zq):
        """Discrete forward prediction evaluating all possible categorical paths."""
        logits = self.transition_model(state_zq)
        # Reshape to [batch, num_actions, num_embeddings]
        return logits.view(-1, self.num_actions, self.num_embeddings)

    def forward(self, curr_image, action=None):
        """VQ-VAE forward pass."""
        z_e, z_q, vq_loss, perplexity, indices = self.encode(curr_image)
        recon = self.decode(z_q)
        return recon, z_e, z_q, vq_loss, perplexity, indices
