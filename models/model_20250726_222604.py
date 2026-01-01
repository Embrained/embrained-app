import torch
from torch import nn

# --- Model & Image Parameters ---
# These are placed here so the model definition is self-contained and
# matches the architecture from the training script.
IMG_H, IMG_W = 120, 160
LATENT_DIM = 32

class Encoder(nn.Module):
    """
    Encodes a high-dimensional image into a low-dimensional latent vector (z).
    This class definition is generated to be a "single source of truth" for
    the encoder architecture, ensuring consistency across different scripts.
    """
    def __init__(self, zdim: int = LATENT_DIM):
        super().__init__()
        # Convolutional layers to extract features
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(256, 1024, 3, 1, 1), nn.LeakyReLU(0.01), nn.MaxPool2d(2)
        )
        # Calculate the flattened dimension after convolutions and pooling
        flat_dim = 1024 * (IMG_H // 16) * (IMG_W // 16)
        
        # Fully connected layers to output the mean and log variance
        self.fc_mu = nn.Linear(flat_dim, zdim)
        self.fc_logvar = nn.Linear(flat_dim, zdim)

    def forward(self, x: torch.Tensor):
        """
        Passes the input image tensor through the network.
        Returns the mean (mu) and log variance (logvar) of the latent space.
        """
        # Flatten the output of the convolutional layers
        h = self.conv(x).reshape(x.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar