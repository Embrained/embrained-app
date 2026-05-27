import argparse
import os
import glob
import math
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from tqdm import tqdm
import sys

# Ensure backend imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.spatial_model import DiscreteVQVAE

def main():
    parser = argparse.ArgumentParser(description="Plot Discrete VAE Active Prototypes")
    parser.add_argument('--model', type=str, default=None, help="Path to VQ-VAE .pth file")
    parser.add_argument('--data', type=str, default='data', help="Path to image data folder")
    parser.add_argument('--max_images', type=int, default=5000, help="Number of random images to sample to find active tokens")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Automatically find latest VQ-VAE if not provided
    if args.model is None:
        candidates = glob.glob(os.path.join(args.data, 'vqvae_*.pth'))
        candidates = [c for c in candidates if 'discrete_cql_model' not in c]
        if not candidates:
            print("No vqvae_*.pth models found in data directory.")
            return
        candidates.sort(key=os.path.getmtime, reverse=True)
        args.model = candidates[0]
    
    print(f"Loading model: {args.model}")
    
    # Parse num_embeddings from filename if possible
    basename = os.path.basename(args.model)
    num_embeddings = 512
    latent_dim = 32
    try:
        parts = basename.replace('.pth', '').split('_')
        num_embeddings = int(parts[1].replace('c', ''))
        latent_dim = int(parts[2].replace('d', ''))
    except:
        pass

    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    model = DiscreteVQVAE(latent_dim=latent_dim, model_size='large', input_spatial_dim=64, in_channels=3, num_embeddings=num_embeddings).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Find active tokens by scanning a subset of the dataset
    print(f"Scanning up to {args.max_images} images to identify active tokens...")
    image_paths = glob.glob(os.path.join(args.data, '**', 'frame_*.jpg'), recursive=True)
    import random
    if len(image_paths) > args.max_images:
        image_paths = random.sample(image_paths, args.max_images)

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    token_counts = Counter()

    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    imgs.append(transform(img))
                except:
                    continue
            if not imgs: continue
            
            batch_tensor = torch.stack(imgs).to(device)
            # Forward pass through encoder and Vector Quantizer
            x_enc = model.encoder(batch_tensor)
            z_e = model.fc_e(x_enc)
            
            flat_input = z_e
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(model.vq.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, model.vq.embedding.weight.t()))
            
            encoding_indices = torch.argmin(distances, dim=1)
            
            for idx in encoding_indices.cpu().numpy():
                token_counts[int(idx)] += 1

    active_tokens = [idx for idx, count in token_counts.most_common()]
    print(f"Found {len(active_tokens)} active tokens out of {num_embeddings} codebook capacity.")

    if not active_tokens:
        print("No active tokens found. Exiting.")
        return

    # Decode the master images
    print("Decoding structural prototypes...")
    prototypes = []
    with torch.no_grad():
        for idx in active_tokens:
            # Grab the dictionary vector directly
            code_vector = model.vq.embedding.weight[idx].unsqueeze(0)
            
            # Decode it
            recon = model.decoder(model.decoder_input(code_vector))
            
            # Convert to image format
            recon_img = recon.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            prototypes.append((idx, token_counts[idx], recon_img))

    # Plotting
    cols = math.ceil(math.sqrt(len(prototypes)))
    rows = math.ceil(len(prototypes) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    import numpy as np
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (idx, count, img) in enumerate(prototypes):
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Token {idx}\nCount: {count}", fontsize=10, pad=4)

    # Turn off remaining empty subplots
    for i in range(len(prototypes), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path = "vqvae_prototypes.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved master templates grid to {output_path}")

if __name__ == "__main__":
    main()
