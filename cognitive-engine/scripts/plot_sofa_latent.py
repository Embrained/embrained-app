"""
Plot sofa examples on a PCA latent-space manifold using the CVE encoder.

Encodes all sofa images and a random sample of ~500 non-sofa exploration
frames, projects the 32-dim latent vectors to 2D via PCA, and saves a
publication-quality scatter plot.
"""

import sys, os, glob, random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CVE_PATH  = os.path.join(_APP_ROOT, 'data', 'cve_32d_20260628_201528.pth')
SOFA_DIR  = os.path.join(_APP_ROOT, 'data', 'sofa')
DATA_DIR  = os.path.join(_APP_ROOT, 'data')
OUT_PATH  = r'C:\Users\chris\Embrained\images\sofa_latent_pca.png'

N_NONSOFA = 500          # how many background frames to sample
BATCH     = 64           # encoding batch size

# ── load encoder ─────────────────────────────────────────────────────────
from modules.spatial_model import TinyVAE, ContrastiveVisuomotorEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")

state_dict = torch.load(CVE_PATH, map_location=device, weights_only=True)
latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(state_dict)
n_actions = state_dict['action_predictor.2.weight'].shape[0]
print(f"[INFO] Detected: latent_dim={latent_dim}, model_size={model_size}, "
      f"img_dim={img_dim}, in_channels={in_channels}, n_actions={n_actions}")

encoder = ContrastiveVisuomotorEncoder(
    latent_dim=latent_dim, model_size=model_size,
    input_spatial_dim=img_dim, in_channels=in_channels,
    n_actions=n_actions
).to(device)
encoder.load_state_dict(state_dict)
encoder.eval()

# ── image transform ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((img_dim, img_dim)),
    transforms.ToTensor(),          # → [0, 1]
])

def load_images(paths):
    """Load a list of image paths → tensor [N, C, H, W]."""
    tensors = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        tensors.append(transform(img))
    return torch.stack(tensors)

# ── gather sofa images ───────────────────────────────────────────────────
sofa_paths = sorted(glob.glob(os.path.join(SOFA_DIR, '*.jpg')))
sofa_basenames = {os.path.basename(p) for p in sofa_paths}
print(f"[INFO] Sofa images: {len(sofa_paths)}")

# ── gather non-sofa images (random sample from markov sessions) ──────────
session_dirs = sorted(glob.glob(os.path.join(DATA_DIR, 'markov_*')))
all_nonsofa = []
for sd in session_dirs:
    img_dir = os.path.join(sd, 'images')
    if not os.path.isdir(img_dir):
        continue
    for fp in glob.glob(os.path.join(img_dir, '*.jpg')):
        if os.path.basename(fp) not in sofa_basenames:
            all_nonsofa.append(fp)

random.seed(42)
nonsofa_paths = random.sample(all_nonsofa, min(N_NONSOFA, len(all_nonsofa)))
print(f"[INFO] Non-sofa pool: {len(all_nonsofa)}, sampled: {len(nonsofa_paths)}")

# ── encode ───────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_paths(paths):
    all_z = []
    for i in range(0, len(paths), BATCH):
        batch = load_images(paths[i:i+BATCH]).to(device)
        z = encoder.encode(batch)
        all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)

print("[INFO] Encoding sofa images …")
z_sofa = encode_paths(sofa_paths)
print("[INFO] Encoding non-sofa images …")
z_nonsofa = encode_paths(nonsofa_paths)

# ── PCA ──────────────────────────────────────────────────────────────────
z_all = np.concatenate([z_sofa, z_nonsofa], axis=0)
pca = PCA(n_components=2)
coords = pca.fit_transform(z_all)

sofa_coords   = coords[:len(z_sofa)]
nonsofa_coords = coords[len(z_sofa):]

print(f"[INFO] PCA explained variance: {pca.explained_variance_ratio_}")

# ── statistics ───────────────────────────────────────────────────────────
sofa_centroid    = z_sofa.mean(axis=0)
nonsofa_centroid = z_nonsofa.mean(axis=0)

centroid_dist = np.linalg.norm(sofa_centroid - nonsofa_centroid)
sofa_dists    = np.linalg.norm(z_sofa - sofa_centroid, axis=1)
sofa_radius   = sofa_dists.mean()
sofa_radius_max = sofa_dists.max()

# Overlap: fraction of non-sofa points within sofa_radius_max of sofa centroid
nonsofa_to_sofa = np.linalg.norm(z_nonsofa - sofa_centroid, axis=1)
overlap_frac = (nonsofa_to_sofa <= sofa_radius_max).mean()

print("\n" + "="*60)
print("LATENT-SPACE STATISTICS  (32-dim, before PCA)")
print("="*60)
print(f"  Sofa cluster centroid -> Non-sofa centroid distance : {centroid_dist:.4f}")
print(f"  Sofa cluster mean radius                          : {sofa_radius:.4f}")
print(f"  Sofa cluster max radius                           : {sofa_radius_max:.4f}")
print(f"  Non-sofa within sofa max-radius (overlap)         : {overlap_frac*100:.1f}%")
print(f"  Sofa examples  : {len(z_sofa)}")
print(f"  Non-sofa sample: {len(z_nonsofa)}")
print("="*60 + "\n")

# ── plot ─────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f8fa')

# Non-sofa: small muted dots
ax.scatter(nonsofa_coords[:, 0], nonsofa_coords[:, 1],
           s=14, c='#b0b0b0', alpha=0.40, edgecolors='none',
           label=f'Non-sofa ({len(z_nonsofa)})')

# Sofa: larger, vivid gradient by distance from sofa centroid (PCA space)
sofa_pca_dists = np.linalg.norm(
    sofa_coords - sofa_coords.mean(axis=0), axis=1)
sc = ax.scatter(sofa_coords[:, 0], sofa_coords[:, 1],
                s=80, c=sofa_pca_dists, cmap='YlOrRd', alpha=0.90,
                edgecolors='#333333', linewidths=0.5,
                label=f'Sofa ({len(z_sofa)})', zorder=5)

cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Distance from sofa centroid (PCA)', fontsize=9, color='#333333')
cbar.ax.tick_params(colors='#555555', labelsize=8)

ax.set_title('CVE Latent Space (PCA) \u2014 Sofa Examples Highlighted',
             fontsize=16, fontweight='bold', pad=14, color='#1a1a1a')
ax.set_xlabel(f'PC-1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
              fontsize=11, color='#333333')
ax.set_ylabel(f'PC-2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
              fontsize=11, color='#333333')

ax.legend(loc='upper right', fontsize=10, framealpha=0.8,
          edgecolor='#cccccc', facecolor='white')

# Annotation box
stats_text = (
    f"Centroid distance: {centroid_dist:.3f}\n"
    f"Sofa mean radius: {sofa_radius:.3f}\n"
    f"Overlap: {overlap_frac*100:.1f}%"
)
ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
        fontsize=9, color='#333333', verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#bbbbbb', alpha=0.9))

ax.tick_params(colors='#444444', labelsize=9)
for spine in ax.spines.values():
    spine.set_color('#cccccc')

fig.tight_layout()

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, facecolor='white', bbox_inches='tight')
print(f"[INFO] Plot saved -> {OUT_PATH}")
plt.close(fig)

