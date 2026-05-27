import os
import sys
import json
import torch
import random
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_DIR
from backend.training.train_fixed_goal import TARGET_IMAGES, normalize_path
from modules.spatial_model import TinyVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = os.path.abspath(DATA_DIR)

def main():
    target_normalized = [normalize_path(p) for p in TARGET_IMAGES]

    vae_candidates = glob.glob(os.path.join(DATA_ROOT, '*-vae_*.pth'))
    vae_candidates = [f for f in vae_candidates if not any(x in f.lower() for x in ['hello_world', 'cql', 'reflex', 'fixed_goal', 'policy'])]
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    VAE_PATH = vae_candidates[0]
    vae_basename = os.path.basename(VAE_PATH).replace('.pth', '')

    vae_state = torch.load(VAE_PATH, map_location=device, weights_only=True)
    latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(vae_state)
    vae = TinyVAE(latent_dim=latent_dim, model_size=model_size, input_spatial_dim=img_dim, in_channels=in_channels).to(device)
    vae.load_state_dict(vae_state)
    vae.eval()

    cache_path = os.path.join(DATA_ROOT, f"{vae_basename}_global_latents.pt")
    latent_dict = {}
    if os.path.exists(cache_path):
        raw_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("path_map", {})
        for k, v in raw_dict.items():
            latent_dict[normalize_path(k)] = v

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    target_latents = []
    with torch.no_grad():
        for target in target_normalized:
            if target in latent_dict:
                target_latents.append(latent_dict[target])
            else:
                img_path = os.path.join(DATA_ROOT, target)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    t_img = transform(img).unsqueeze(0).to(device)
                    _, mu, _ = vae(t_img)
                    target_latents.append(mu.cpu().squeeze())
                    latent_dict[target] = mu.cpu().squeeze()

    target_latents = torch.stack(target_latents) # [10, 32]
    
    trans_path = os.path.join(DATA_ROOT, "all_transitions.json")
    with open(trans_path, 'r') as f:
        all_data = json.load(f)

    candidate_images = []
    visited = set()
    for node in all_data:
        p = normalize_path(node.get('image_path', ''))
        if p and p not in visited:
            visited.add(p)
            if p in latent_dict:
                z = latent_dict[p]
                dists = torch.norm(target_latents - z.unsqueeze(0), dim=1)
                min_dist = torch.min(dists).item()
                if min_dist <= 1.50:
                    candidate_images.append((p, min_dist))

    # We want exactly 30 candidates for a 6x5 grid
    if len(candidate_images) > 30:
        sampled = random.sample(candidate_images, 30)
    else:
        sampled = candidate_images

    # Sort internally by distance just to make reading the grid nicer
    sampled.sort(key=lambda x: x[1])

    fig, axes = plt.subplots(6, 5, figsize=(15, 18))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(sampled):
            img_path, dist = sampled[i]
            abs_path = os.path.join(DATA_ROOT, img_path)
            if os.path.exists(abs_path):
                img = Image.open(abs_path)
                ax.imshow(img)
                ax.set_title(f"Dist: {dist:.2f}", fontsize=10, color='darkgreen')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.suptitle("Candidate Target Images (Latent Distance <= 1.50)", fontsize=16)

    # Save globally and locally
    artifacts_path = os.path.abspath(r"C:\Users\chris\.gemini\antigravity\brain\579c23c7-abeb-4e15-acc5-06311e12eb85\goal_candidates_grid.png")
    os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
    plt.savefig(artifacts_path, dpi=150, bbox_inches='tight')

    save_path = os.path.join(DATA_ROOT, "goal_candidates_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("DONE! Plot saved successfully.")

if __name__ == "__main__":
    main()
