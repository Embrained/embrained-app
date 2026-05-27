import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import random
import sys

sys.path.insert(0, r'c:\Users\chris\Embrained\software_suite')
from modules.spatial_model import DiscreteVQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = r'C:\Users\chris\Embrained\software_suite\data'

# 1. Load VAE
vae_path = r'C:\Users\chris\Embrained\software_suite\data\vqvae_512c_32d_20260428_130632.pth'
state_dict = torch.load(vae_path, map_location=device, weights_only=True)
codebook = state_dict['vq.embedding.weight']  # [512, 32]

# 2. Load exact latent
exact_latent_path = os.path.join(DATA_ROOT, 'goals', 'discrete_exact_stats.json')
with open(exact_latent_path, 'r') as f:
    stats = json.load(f)
exact_latent = torch.tensor(stats['exact_latent'], dtype=torch.float32).to(device)

# Hardcode tokens ranked 6-10
all_target_tokens = [112, 200, 139, 352, 177]

print(f"Target tokens: {all_target_tokens}")

# 3. Process dataset to assign tokens
json_path = os.path.join(DATA_ROOT, "all_transitions.json")
with open(json_path, 'r') as f:
    all_data = json.load(f)

vae = DiscreteVQVAE(latent_dim=32, model_size='large', input_spatial_dim=64, in_channels=3, num_embeddings=512).to(device)
vae.load_state_dict(state_dict)
vae.eval()

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
])

token_images = {k: [] for k in all_target_tokens}

random.shuffle(all_data)

print("Processing images to find token matches...")
for i in range(0, len(all_data), 128):
    batch_data = all_data[i:i+128]
    imgs = []
    valid_paths = []
    for item in batch_data:
        p = item.get('image_path', '')
        if not os.path.isabs(p):
            p = os.path.join(DATA_ROOT, p)
        if 'webcam_frame_' not in p and 'frame_' in os.path.basename(p) and os.path.exists(p):
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img))
            valid_paths.append(p)
            
    if not imgs:
        continue
        
    imgs_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        x_enc = vae.encoder(imgs_tensor)
        z_e = vae.fc_e(x_enc)
        
        # Calculate distances to codebook
        d = torch.sum(z_e ** 2, dim=-1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - \
            2 * torch.matmul(z_e, codebook.t())
        indices = torch.argmin(d, dim=-1).cpu().numpy()
        
    for idx, token_id in enumerate(indices):
        if token_id in token_images:
            token_images[token_id].append(valid_paths[idx])
            
    # Check if we have at least 6 for all
    all_done = True
    for k, v in token_images.items():
        if len(v) < 6:
            all_done = False
            break
            
    if all_done:
        print("Found 6 images for all required tokens!")
        break
        
    if i % 1280 == 0:
        counts = {k: len(v) for k, v in token_images.items()}
        print(f"Processed {i} images... Counts: {counts}")

# Plot
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
plt.subplots_adjust(wspace=0.05, hspace=0.3)

for row_idx, token_id in enumerate(all_target_tokens):
    paths = token_images[token_id]
    if len(paths) >= 6:
        selected_paths = random.sample(paths, 6)
    else:
        selected_paths = paths + [None] * (6 - len(paths))
        
    for col_idx, path in enumerate(selected_paths):
        ax = axes[row_idx, col_idx]
        if path is not None:
            img = Image.open(path)
            ax.imshow(img)
            title = f"Rank {row_idx+6} Token {token_id}"
            ax.set_title(title, fontsize=10)
        ax.axis('off')

out_path = os.path.join(DATA_ROOT, 'goal_neighborhood_grid_6_10.png')
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"Saved to {out_path}")
