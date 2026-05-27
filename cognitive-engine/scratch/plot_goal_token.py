import os
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.spatial_model import DiscreteVQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
vae_path = os.path.join(data_root, 'vqvae_512c_32d_20260427_153402.pth')
stats_path = os.path.join(data_root, 'goals', 'discrete_exact_stats.json')
transitions_path = os.path.join(data_root, 'all_transitions.json')

# 1. Load the Discrete VQ-VAE
state_dict = torch.load(vae_path, map_location=device, weights_only=True)
model = DiscreteVQVAE(latent_dim=32, model_size='large', input_spatial_dim=64, in_channels=3, num_embeddings=512).to(device)
model.load_state_dict(state_dict)
model.eval()

# 2. Load Exact Latent Vector
with open(stats_path, 'r') as f:
    stats = json.load(f)
exact_latent = torch.tensor(stats['exact_latent'], dtype=torch.float32).to(device)

# 3. Find the exact Token Index
# Measure distance from exact_latent to all 512 codebook vectors
with torch.no_grad():
    exact_dist = torch.sum(exact_latent ** 2) + \
                 torch.sum(model.vq.embedding.weight ** 2, dim=1) - \
                 2 * torch.matmul(exact_latent.unsqueeze(0), model.vq.embedding.weight.t())
    exact_idx = torch.argmin(exact_dist, dim=1)[0].item()

print(f"The Goal Token is Index: {exact_idx}")

# 4. Search for 30 images matching this token
with open(transitions_path, 'r') as f:
    transitions = json.load(f)

# Extract all unique image paths
image_paths = list(set([t['image_path'] for t in transitions]))
random.shuffle(image_paths)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

print(f"Searching for 30 images matching Token {exact_idx}...")
matching_images = []

for img_rel_path in image_paths:
    img_path = os.path.join(data_root, img_rel_path)
    if not os.path.exists(img_path):
        continue
        
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            x_enc = model.encoder(img_tensor)
            z_e = model.fc_e(x_enc)
            d = torch.sum(z_e ** 2, dim=1, keepdim=True) + torch.sum(model.vq.embedding.weight ** 2, dim=1) - 2 * torch.matmul(z_e, model.vq.embedding.weight.t())
            idx = torch.argmin(d, dim=1)[0].item()
            
        if idx == exact_idx:
            matching_images.append(img_pil)
            print(f"Found {len(matching_images)}/30...", end='\r')
            
            if len(matching_images) >= 30:
                print("\nFound 30 matching images!")
                break
    except Exception as e:
        continue

# 5. Plot the Grid
if len(matching_images) > 0:
    n_imgs = len(matching_images)
    cols = min(6, n_imgs)
    rows = (n_imgs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    if n_imgs == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle(f"Goal Token: {exact_idx} (Discrete VQ-VAE) - {n_imgs} found", fontsize=20)
    
    for i, ax in enumerate(axes):
        if i < len(matching_images):
            ax.imshow(matching_images[i])
        ax.axis('off')
        
    plt.tight_layout()
    out_img = os.path.join(data_root, 'goals', 'goal_token_grid.png')
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"Saved Goal Token Image Grid to {out_img}")
else:
    print("Could not find ANY matching images.")
