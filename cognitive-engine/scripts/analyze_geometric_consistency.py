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


import os
import sys
import json
import glob
import logging
import random
import base64
import io
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import torchvision.transforms as T

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GeoConsistency")

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.spatial_model import TinyVAE, CQLNetwork
from config import HIDDEN_DIM

# --- Configuration ---
DATASET_NAME = "nook"
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", DATASET_NAME)
VAE_MODEL_PATH = os.path.join(DATA_ROOT, f"{DATASET_NAME}-vae.pth")
POL_MODEL_PATH = os.path.join(DATA_ROOT, f"{DATASET_NAME}-vae-cql.pth") # Default naming convention
OUTPUT_DIR = os.getcwd()

IMG_H = 64
IMG_W = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
])

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return torch.zeros((3, IMG_H, IMG_W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)

class CustomVAE(nn.Module):
    """
    Custom VAE to match legacy checkpoints (e.g. 1024 flatten dim / Enormous).
    Re-engineered: Base 64, Layers 6, Cap 1024.
    """
    def __init__(self, latent_dim=32, base_channels=64, n_layers=6):
        super(CustomVAE, self).__init__()
        self.base_channels = base_channels
        self.n_layers = n_layers
        self.latent_dim = latent_dim

        modules = []
        in_channels = 3
        current_channels = self.base_channels
        
        # 1. Initial Conv
        modules.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, stride=1, padding=1))
        modules.append(nn.ReLU())
        
        # 2. Downsampling Stack
        for i in range(self.n_layers):
            out_channels = min(current_channels * 2, 1024) 
            modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            modules.append(nn.ReLU())
            current_channels = out_channels
            
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)
        
        final_spatial = 64 // (2 ** self.n_layers)
        self.flatten_dim = current_channels * final_spatial * final_spatial
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        dec_modules = []
        dec_modules.append(nn.Unflatten(1, (current_channels, final_spatial, final_spatial)))
        
        for i in range(self.n_layers):
            is_last = (i == self.n_layers - 1)
            target_out = current_channels // 2 if not is_last else 3
            
            if is_last:
                dec_modules.append(nn.ConvTranspose2d(current_channels, 3, kernel_size=4, stride=2, padding=1))
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

def load_models():
    """Loads VAE and CQL Policy with auto-detection of sizes."""
    logger.info(f"Loading VAE from {VAE_MODEL_PATH}...")
    if not os.path.exists(VAE_MODEL_PATH):
        raise FileNotFoundError(f"VAE model not found at {VAE_MODEL_PATH}")

    # Load VAE State
    vae_state = torch.load(VAE_MODEL_PATH, map_location=DEVICE)
    
    # 1. Detect VAE Size/Latent Dim
    latent_dim = 32
    model_size = 'small'
    use_custom = False
    
    if 'fc_mu.weight' in vae_state:
        weight_shape = vae_state['fc_mu.weight'].shape
        latent_dim = weight_shape[0]
        flatten_dim = weight_shape[1]
        
        logger.info(f"Detected VAE: LatentDim={latent_dim} (Flatten={flatten_dim})")
        
        if flatten_dim == 1024:
            logger.info("Handling Legacy VAE (Flatten 1024)... trying Base=16, Layers=5")
            use_custom = True
        elif flatten_dim == 4096:
            model_size = 'tiny'
        elif flatten_dim == 8192:
            model_size = 'small'
        elif flatten_dim == 16384:
            model_size = 'medium'
        else:
            model_size = 'large'

    if use_custom:
        encoder = CustomVAE(latent_dim=latent_dim, base_channels=16, n_layers=6).to(DEVICE)
    else:
        encoder = TinyVAE(latent_dim=latent_dim, model_size=model_size).to(DEVICE)
        
    encoder.load_state_dict(vae_state)
    encoder.eval()
    
    # 2. Load Policy
    # 2. Load Policy
    logger.info(f"Loading Policy from {POL_MODEL_PATH}...")
    if not os.path.exists(POL_MODEL_PATH):
        # Try fallback name
        fallback = os.path.join(DATA_ROOT, "cql_policy.pth")
        if os.path.exists(fallback):
            logger.info(f"Using fallback policy: {fallback}")
            pol_path = fallback
        else:
            raise FileNotFoundError(f"Policy model not found at {POL_MODEL_PATH}")
    else:
        pol_path = POL_MODEL_PATH

    policy_state = torch.load(pol_path, map_location=DEVICE)
    
    # Extract weights if saved as dict with valid_actions
    if isinstance(policy_state, dict) and 'model_state_dict' in policy_state:
        policy_state = policy_state['model_state_dict']
        
    # [NEW] Auto-Detect Policy Architecture from Weights
    # input_layer.weight shape is (hidden_dim, input_dim)
    if 'input_layer.weight' in policy_state:
        w_shape = policy_state['input_layer.weight'].shape
        detected_hidden = w_shape[0]
        detected_input = w_shape[1]
        
        logger.info(f"Detected Policy Architecture: Hidden={detected_hidden}, Input={detected_input}")
        
        # Map to Size
        if detected_hidden == 256:
            pol_size = 'small'
        elif detected_hidden == 512:
            pol_size = 'medium'
        elif detected_hidden == 1024:
            pol_size = 'large'
        else:
            pol_size = 'large' # Fallback
            
        # Instantiate
        # Note: If detected_input != latent_dim * 2, we have a mismatch but we must match the POLICY to load it.
        policy = CQLNetwork(input_dim=detected_input, hidden_dim=detected_hidden, action_dim=5, model_size=pol_size).to(DEVICE)
        
        try:
            policy.load_state_dict(policy_state)
            logger.info(f"Successfully loaded {pol_size} Policy.")
        except RuntimeError as e:
            logger.error(f"Failed to load despite detection: {e}")
            raise e
            
        # Return mismatch info if needed
        policy_expected_latent = detected_input // 2
        
        return encoder, policy, policy_expected_latent
        
    else:
        # Fallback to trial and error if weird format
        logger.warning("Could not detect policy architecture. Trying defaults...")
        
        policy_input_dim = latent_dim * 2
        policy = CQLNetwork(input_dim=policy_input_dim, hidden_dim=HIDDEN_DIM, action_dim=5, model_size='large').to(DEVICE)
    
        try:
            policy.load_state_dict(policy_state)
        except RuntimeError as e:
             logger.warning(f"Default Policy size 'large' failed. Trying 'medium'...")
             try:
                  policy = CQLNetwork(input_dim=policy_input_dim, hidden_dim=512, action_dim=5, model_size='medium').to(DEVICE)
                  policy.load_state_dict(policy_state)
                  logger.info("Loaded 'medium' policy.")
             except RuntimeError:
                  logger.warning("Trying 'small' policy...")
                  policy = CQLNetwork(input_dim=policy_input_dim, hidden_dim=256, action_dim=5, model_size='small').to(DEVICE)
                  policy.load_state_dict(policy_state)
                  logger.info("Loaded 'small' policy.")

        policy.eval()
        
        return encoder, policy, latent_dim

def get_all_image_paths():
    ep_path = os.path.join(DATA_ROOT, "episodes.json")
    if not os.path.exists(ep_path):
        raise FileNotFoundError(f"episodes.json not found in {DATA_ROOT}")
        
    with open(ep_path, 'r') as f:
        episodes = json.load(f)
        
    paths = []
    for ep in episodes:
        # Include start frame
        paths.append(os.path.join(DATA_ROOT, ep['start_frame']['image_path']))
        for step in ep['actions']:
             paths.append(os.path.join(DATA_ROOT, step['image_path']))
             
    # Deduplicate
    paths = list(set(paths))
    logger.info(f"Found {len(paths)} unique images.")
    return paths

def main():
    logger.info("Starting Geometric Consistency Analysis...")
    
    # 0. Load Resources
    encoder, policy, latent_dim = load_models()
    image_paths = get_all_image_paths()
    
    # Subsample if too many (for speed)
    if len(image_paths) > 2000:
        logger.info("Subsampling to 2000 images for manifold generation...")
        eval_paths = random.sample(image_paths, 2000)
    else:
        eval_paths = image_paths

    # 1. Manifold Linearization
    logger.info("Step 1: Manifold Linearization...")
    latents = []
    
    # Batch process
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(eval_paths), batch_size):
            batch_paths = eval_paths[i:i+batch_size]
            tensors = [load_image(p) for p in batch_paths]
            batch_tensor = torch.stack(tensors).to(DEVICE)
            
            # Encoder returns (recon, mu, logvar) -> use mu
            _, mu, _ = encoder(batch_tensor)
            latents.append(mu.cpu().numpy())
            
    latents = np.concatenate(latents, axis=0)
    
    # PCA
    logger.info("Fitting PCA...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents) # (N, 2)
    
    # Compute Phase
    # arctan2(y, x) -> [-pi, pi]
    phases = np.arctan2(latents_2d[:, 1], latents_2d[:, 0])
    
    # Normalize to [0, 2pi) optionally, but -pi to pi is fine for distance calc
    
    # Save/Visualize PCA ring
    plt.figure(figsize=(6,6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=phases, cmap='hsv', s=5, alpha=0.5)
    plt.title("Manifold Phase Visualization")
    plt.colorbar(label="Phase (rad)")
    plt.savefig("manifold_pca_phase.png")
    logger.info("Saved manifold_pca_phase.png")

    # 2. Q-Phase Correlation Loop
    logger.info("Step 2: Q-Phase Correlation Loop...")
    
    N_PAIRS = 500
    results_dtheta = []
    results_advantage = []
    correct_predictions = 0
    valid_pairs = 0
    
    # Prepare batch of pairs? No, loop is fine for 500
    
    for _ in range(N_PAIRS):
        # Pick two random indices from our latent set
        idx1, idx2 = np.random.choice(len(latents), 2, replace=False)
        
        phi1 = phases[idx1]
        phi2 = phases[idx2]
        
        # Ground Truth Distance (angular)
        d_theta = phi2 - phi1
        # Wrap to [-pi, pi]
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Determine Goal Direction
        # If d_theta > 0: Counter-Clockwise (Left?) or Right?
        # User Context: "Metric: If d_theta in (0, pi), goal is CCW (Left). If (-pi, 0), goal is CW (Right)."
        # Note: This mapping depends on how PCA projects and how the robot turned. 
        # Typically Left turn = yaw increases = CCW.
        
        if abs(d_theta) < 0.1: # Too close, ignore for accuracy metric (Stop zone)
            is_stop_zone = True
        else:
            is_stop_zone = False
            
        # [UPDATED] Logic Inversion per user request
        # Previous: >0 Left, <0 Right
        # New: >0 Right, <0 Left
        target_action = "RIGHT" if d_theta > 0 else "LEFT"
        
        # Query Q-Network
        path1 = eval_paths[idx1]
        path2 = eval_paths[idx2] # Goal
        
        t1 = load_image(path1).unsqueeze(0).to(DEVICE)
        t2 = load_image(path2).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
             # Get latents again? No, we have latents, but Q-net wrapper typically takes images or latents.
             # In train_cql.py, FullQNet takes (img, img).
             
             # But wait, Q-Net inside constructs latent from image.
             # Let's use the loaded encoder to get latents (already done?) -> No, FullQNet does it internally.
             # But we can just construct the latent input vector manually for efficiency if we use policy directly.
             
             z1 = torch.tensor(latents[idx1]).unsqueeze(0).to(DEVICE)
             z2 = torch.tensor(latents[idx2]).unsqueeze(0).to(DEVICE)
             
             # [Handle Mismatch] If VAE gave 64 but Policy wants 32
             if z1.shape[1] > latent_dim:
                 z1 = z1[:, :latent_dim]
                 z2 = z2[:, :latent_dim]
             
             state_input = torch.cat([z1, z2], dim=1) # (1, 2*L)
             q_values = policy(state_input) # (1, 5)
             
        q_vals = q_values.cpu().numpy()[0]
        # Actions: 0:Fwd, 1:Left, 2:Right, 3:Stop, 4:Back
        q_left = q_vals[1]
        q_right = q_vals[2]
        q_stop = q_vals[3]
        
        advantage = q_left - q_right
        
        results_dtheta.append(d_theta)
        results_advantage.append(advantage)
        
        # Check agreement
        if not is_stop_zone:
            valid_pairs += 1
            if d_theta > 0: # Target RIGHT (Previous Left)
                if q_right > q_left: correct_predictions += 1
            else: # Target LEFT (Previous Right)
                if q_left > q_right: correct_predictions += 1
                
    accuracy = (correct_predictions / valid_pairs) * 100 if valid_pairs > 0 else 0
    logger.info(f"Geodesic Agreement Accuracy: {accuracy:.2f}% (on {valid_pairs} pairs)")

    # 3. Visualizations & Metrics
    logger.info("Step 3: Generating Plots...")
    
    # Plot A: Sigmoid Check
    plt.figure(figsize=(8, 6))
    plt.scatter(results_dtheta, results_advantage, alpha=0.6, c='blue', edgecolors='none')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Q-Advantage (Left - Right) vs Angular Distance")
    plt.xlabel("Signed Angular Distance (rad)")
    plt.ylabel("Advantage (Q_Left - Q_Right)")
    plt.grid(True, alpha=0.3)
    plt.savefig("q_phase_sigmoid.png")
    logger.info("Saved q_phase_sigmoid.png")
    
    # Plot B: Decision Boundary Heatmap
    # Grid of Start Angle vs Goal Angle
    res = 50
    start_phis = np.linspace(-np.pi, np.pi, res)
    goal_phis = np.linspace(-np.pi, np.pi, res)
    
    grid_map = np.zeros((res, res))
    
    # For heatmap, we need to map Angle -> Latent
    # We can't easily inverse PCA.
    # Approach: For each grid cell (phi_s, phi_g), find NEAREST real data points in manifold.
    
    # Build a KDTree or just simple nearest search
    # Since res=50x50=2500, we can do brute force nearest for the prototype.
    
    logger.info("Generating Heatmap (finding nearest neighbors)...")
    
    for i, p_s in enumerate(start_phis):
        # Find nearest latent to p_s
        # Circular distance
        diffs_s = np.arctan2(np.sin(phases - p_s), np.cos(phases - p_s))
        idx_s = np.argmin(np.abs(diffs_s))
        z_s = torch.tensor(latents[idx_s]).unsqueeze(0).to(DEVICE)
        
        for j, p_g in enumerate(goal_phis):
            diffs_g = np.arctan2(np.sin(phases - p_g), np.cos(phases - p_g))
            idx_g = np.argmin(np.abs(diffs_g))
            z_g = torch.tensor(latents[idx_g]).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # [Handle Mismatch]
                if z_s.shape[1] > latent_dim:
                    z_s = z_s[:, :latent_dim]
                if z_g.shape[1] > latent_dim:
                    z_g = z_g[:, :latent_dim]
                    
                inp = torch.cat([z_s, z_g], dim=1)
                q = policy(inp).cpu().numpy()[0]
                
            # Color encoding: 1=Left(Blue), -1=Right(Red), 0=Stop(White)
            # Actions: 1:Left, 2:Right, 3:Stop
            best_act = np.argmax(q)
            
            val = 0
            if best_act == 1: val = 1 # Left (Blue)
            elif best_act == 2: val = -1 # Right (Red)
            elif best_act == 3: val = 0 # Stop (White)
            
            grid_map[j, i] = val # y=goal, x=start
            
    # Plot Heatmap
    plt.figure(figsize=(7, 7))
    # Custom cmap: Red(-1) -> White(0) -> Blue(1)
    from matplotlib.colors import LinearSegmentedColormap
    # bwr is Blue-White-Red. We want Red-White-Blue? 
    # bwr: Blue=Low, Red=High.
    # We mapped Right=-1, Left=1. 
    # If we use bwr: -1(Right) is Blue? No.
    # Let's define explicitly.
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)] # R -> W -> B
    cmap = LinearSegmentedColormap.from_list("custom_policy", colors, N=3)
    
    plt.imshow(grid_map, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], cmap='bwr', alpha=0.8) 
    # 'bwr' standard: Blue(low) to Red(high). 
    # My val: -1 (Right), 1 (Left).
    # So 'bwr' -> Right=Blue, Left=Red. 
    # User wanted: Blue=Left, Red=Right.
    # So I need Low=Red, High=Blue. -> Use 'seismic_r' or just reverse bwr? 'bwr_r'?
    # Or just construct:
    cmap_rb = LinearSegmentedColormap.from_list("rb", ["red", "white", "blue"])
    
    plt.imshow(grid_map, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap_rb)
    
    plt.colorbar(ticks=[-1, 0, 1], label="Action (Right | Stop | Left)")
    plt.xlabel("Start Angle")
    plt.ylabel("Goal Angle")
    plt.title("Policy Decision Boundary")
    plt.savefig("policy_topology_heatmap.png")
    logger.info("Saved policy_topology_heatmap.png")
    
    print("-" * 30)
    print(f"Geodesic Agreement Accuracy: {accuracy:.2f}%")
    print("-" * 30)
    
    # 4. Generate HTML Report
    generate_html_report(accuracy, valid_pairs, latent_dim, policy.hidden_dim if hasattr(policy, 'hidden_dim') else 'Unknown')

def generate_html_report(accuracy, n_pairs, latent_dim, policy_hidden):
    def img_to_base64(path):
        if not os.path.exists(path): return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    img_manifold = img_to_base64("manifold_pca_phase.png")
    img_sigmoid = img_to_base64("q_phase_sigmoid.png")
    img_heatmap = img_to_base64("policy_topology_heatmap.png")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Geometric Consistency Analysis</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background: #f4f4f4; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .metric {{ font-size: 1.2em; font-weight: bold; color: #007bff; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .card {{ border: 1px solid #ddd; padding: 10px; border-radius: 4px; text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
            .footer {{ margin-top: 20px; font-size: 0.9em; color: #666; }}
            code {{ background: #eee; padding: 2px 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Geometric Consistency Analysis</h1>
            
            <p class="metric">Geodesic Agreement Accuracy: {accuracy:.2f}%</p>
            <p><strong>Evaluated Pairs:</strong> {n_pairs}</p>
            <p><strong>Model Config:</strong> VAE Latent={latent_dim}, Policy Hidden={policy_hidden}</p>
            
            <div class="grid">
                <div class="card">
                    <h3>1. Manifold Phase (Ground Truth)</h3>
                    <img src="data:image/png;base64,{img_manifold}" />
                    <p>Latent ring topology extracted via PCA.</p>
                </div>
                <div class="card">
                    <h3>2. Q-Advantage vs Distance</h3>
                    <img src="data:image/png;base64,{img_sigmoid}" />
                    <p>Expected: Sigmoid. Actual: Policy preference.</p>
                </div>
                <div class="card" style="grid-column: span 2;">
                    <h3>3. Policy Decision Boundary</h3>
                    <img src="data:image/png;base64,{img_heatmap}" />
                    <p>Decision map (Red=Right, Blue=Left) over Start/Goal angles.</p>
                </div>
            </div>
            
            <div class="footer">
                <p>Run locally: <code>python scripts/analyze_geometric_consistency.py</code></p>
                <p>Generated by Embrained-AI</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("geometric_consistency_report.html", "w") as f:
        f.write(html)
        
    logger.info("Saved geometric_consistency_report.html")
    print(f"Report generated: {os.path.abspath('geometric_consistency_report.html')}")

if __name__ == "__main__":
    main()
