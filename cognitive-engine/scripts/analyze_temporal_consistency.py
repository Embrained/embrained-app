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
import cv2
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TempConsistency")

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.spatial_model import TinyVAE, CQLNetwork
from config import HIDDEN_DIM

# --- Configuration ---
DATASET_NAME = "nook"
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", DATASET_NAME)
VAE_MODEL_PATH = os.path.join(DATA_ROOT, f"{DATASET_NAME}-vae.pth")
POL_MODEL_PATH = os.path.join(DATA_ROOT, f"{DATASET_NAME}-vae-cql.pth")
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
    Custom VAE to match legacy checkpoints.
    Base 16, Layers 6.
    """
    def __init__(self, latent_dim=32, base_channels=16, n_layers=6, max_channels=1024):
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
            out_channels = min(current_channels * 2, max_channels) 
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
        for i in range(self.n_layers):
            is_last = (i == self.n_layers - 1)
            
            # Simple halving
            target_out = current_channels // 2
            
            # [Fix] Enforce floor at base_channels to match checkpoint
            if target_out < self.base_channels:
                target_out = self.base_channels
                
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
    """Loads VAE and CQL Policy with robust auto-detection."""
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
            custom_layers = 5
        elif flatten_dim == 2048:
             # Base 64, Layers 5, Max 512 -> 512 * 4 (2x2) = 2048
             logger.info("Handling Legacy VAE (Flatten 2048)... trying Base=64, Layers=5, Max=512")
             use_custom = True
             custom_layers = 5
             custom_base = 64
             custom_max = 512
        elif flatten_dim == 4096:
            model_size = 'tiny'
        elif flatten_dim == 8192:
            model_size = 'small'
        elif flatten_dim == 16384:
            model_size = 'medium'
        elif flatten_dim == 32768:
            model_size = 'large'
        else:
            model_size = 'large'

    if use_custom:
        # Default or Specifics
        c_base = locals().get('custom_base', 16)
        c_layers = locals().get('custom_layers', 6)
        c_max = locals().get('custom_max', 1024)
        
        encoder = CustomVAE(latent_dim=latent_dim, base_channels=c_base, n_layers=c_layers, max_channels=c_max).to(DEVICE)
    else:
        encoder = TinyVAE(latent_dim=latent_dim, model_size=model_size).to(DEVICE)
        
    try:
        encoder.load_state_dict(vae_state)
    except Exception as e:
        logger.warning(f"Default VAE load failed: {e}. Attempting legacy fallback (Base 16, Layers 5)...")
        encoder = CustomVAE(latent_dim=latent_dim, base_channels=16, n_layers=5).to(DEVICE)
        encoder.load_state_dict(vae_state)

    encoder.eval()
    
    # 2. Load Policy
    logger.info(f"Loading Policy from {POL_MODEL_PATH}...")
    if not os.path.exists(POL_MODEL_PATH):
        fallback = os.path.join(DATA_ROOT, "cql_policy.pth")
        if os.path.exists(fallback):
             pol_path = fallback
        else:
            raise FileNotFoundError(f"Policy model not found at {POL_MODEL_PATH}")
    else:
        pol_path = POL_MODEL_PATH

    policy_state = torch.load(pol_path, map_location=DEVICE)
    if isinstance(policy_state, dict) and 'model_state_dict' in policy_state:
        policy_state = policy_state['model_state_dict']
        
    # Auto-Detect Policy Architecture
    if 'input_layer.weight' in policy_state:
        w_shape = policy_state['input_layer.weight'].shape
        detected_hidden = w_shape[0]
        detected_input = w_shape[1]
        
        if detected_hidden == 256: pol_size = 'small'
        elif detected_hidden == 512: pol_size = 'medium'
        elif detected_hidden == 1024: pol_size = 'large'
        else: pol_size = 'large'
            
        policy = CQLNetwork(input_dim=detected_input, hidden_dim=detected_hidden, action_dim=5, model_size=pol_size).to(DEVICE)
        policy.load_state_dict(policy_state)
        logger.info(f"Loaded {pol_size} Policy (Input={detected_input}).")
        
        return encoder, policy, latent_dim, detected_input
    else:
        raise RuntimeError("Could not detect policy architecture.")

def main():
    logger.info("Starting Temporal Consistency Analysis...")
    
    # 1. Load Models
    encoder, policy, latent_dim, policy_input_dim = load_models()
    
    # --- 1. Data Loading (from all_transitions.json) ---
    transitions_path = os.path.join(DATA_ROOT, 'all_transitions.json')
    if not os.path.exists(transitions_path):
        logger.error(f"all_transitions.json not found at {transitions_path}")
        return

    logger.info(f"Loading transitions from {transitions_path}...")
    with open(transitions_path, 'r') as f:
        all_data = json.load(f)

    # Group by session to form trajectories
    sessions = {}
    for item in all_data:
        sess = item['session']
        if sess not in sessions:
            sessions[sess] = []
        sessions[sess].append(item)

    # Sort each session by timestamp
    trajectories = []
    for sess_id, items in sessions.items():
        sorted_items = sorted(items, key=lambda x: x['timestamp'])
        trajectories.append(sorted_items)
        logger.info(f"Session {sess_id}: {len(sorted_items)} steps")

    logger.info(f"Loaded {len(trajectories)} trajectories.")

    # --- 2. Sampling Loop ---
    logger.info("Sampling 1000 homogeneous trajectory segments...")
    valid_samples = []
    attempts = 0
    max_attempts = 100000

    def discretize_action(left, right):
        """
        Maps motor commands to Action IDs (MATCHING comms.py/train_cql.py!).
        Phys FWD  (0) <- l:-S, r:S
        Phys LEFT (1) <- l:-S, r:-S (Turning Left)
        Phys RIGHT(2) <- l:S, r:S   (Turning Right)
        Phys STOP (3) <- l:0, r:0
        Phys BACK (4) <- l:S, r:-S
        """
        tol = 40 # UPDATED: Strict threshold to ignore deadband noise
        if abs(left) < 1 and abs(right) < 1: return 3 # STOP
        if left < -tol and right > tol: return 0 # FWD
        if left < -tol and right < -tol: return 1 # LEFT
        if left > tol and right > tol: return 2 # RIGHT
        if left > tol and right < -tol: return 4 # BACK
        return 3 # Default

    while len(valid_samples) < 1000 and attempts < max_attempts:
        attempts += 1
        
        # Pick random trajectory
        traj = random.choice(trajectories)
        if len(traj) < 25: continue

        # Pick Start t
        t = random.randint(0, len(traj) - 21) # Ensure space for max k=20
        
        # Pick k (5 to 20)
        k = random.randint(5, 20)
        
        # Check Homogeneity (Ignoring STOPs)
        direction = None
        is_valid = True
        has_movement = False
        
        for i in range(t, t+k):
            step = traj[i]
            l = step.get('left_cmd', 0)
            r = step.get('right_cmd', 0)
            act_id = discretize_action(l, r)
            
            if act_id == 3: # STOP
                continue # Ignore stops
                
            # We only allow turning actions (1 or 2) for this analysis
            if act_id not in [1, 2]:
                is_valid = False
                break
                
            has_movement = True
            if direction is None:
                direction = act_id
            elif act_id != direction:
                is_valid = False
                break
        
        if is_valid and has_movement:
            # S_t is image at index t
            # S_{t+k} is image at index t+k
            start_path = os.path.join(DATA_ROOT, traj[t]['image_path'])
            goal_path = os.path.join(DATA_ROOT, traj[t+k]['image_path'])
            
            valid_samples.append({
                'start': start_path,
                'goal': goal_path,
                'gt': direction, # 1 or 2
                'dt': k
            })

    logger.info(f"Collected {len(valid_samples)} valid samples.")
    if len(valid_samples) == 0:
        logger.error("Failed to collect any valid samples. Aborting.")
        return
    
    # 3. Inference Loop
    y_true = []
    y_pred = []
    
    correct = 0
    
    logger.info("Running Inference...")
    with torch.no_grad():
        for sample in valid_samples:
            img_s = load_image(sample['start']).unsqueeze(0).to(DEVICE)
            img_g = load_image(sample['goal']).unsqueeze(0).to(DEVICE)
            
            # Encode
            _, mu_s, _ = encoder(img_s)
            _, mu_g, _ = encoder(img_g)
            
            # Handle dim mismatch if policy still expects 2*32 but VAE is 64?
            # User said fully updated, but let's be safe.
            # load_models returns detected policy_input_dim.
            
            # If policy wants 64 input, it implies 32 latent.
            # If we detect latent=32, great.
            # If we detect latent=64, but policy input 64, we must slice.
            
            if mu_s.shape[1] * 2 > policy_input_dim:
                # Slice
                target_lat = policy_input_dim // 2
                mu_s = mu_s[:, :target_lat]
                mu_g = mu_g[:, :target_lat]
                
            state_input = torch.cat([mu_s, mu_g], dim=1)
            
            q_values = policy(state_input)
            q_best = torch.argmax(q_values, dim=1).item()
            
            y_true.append(sample['gt'])
            y_pred.append(q_best)
            
            if q_best == sample['gt']:
                correct += 1
                
    accuracy = (correct / len(valid_samples)) * 100
    
    # 4. Visualization (Confusion Matrix)
    labels = [1, 2, 3] # Left, Right, Stop (Include Stop to see failure mode)
    label_names = ["LEFT", "RIGHT", "STOP"]
    
    # Map any other preds (0, 4) to 'Other' or just ignore in labels?
    # Simple logic: map pred->label idx
    
    logger.info(f"Final Accuracy: {accuracy:.2f}% ({correct}/{len(y_true)})")
    
    # [NEW] Debug Logging
    # from sklearn.metrics import confusion_matrix # Already imported implicitly
    import numpy as np
    
    # Use all possible action labels for the confusion matrix logging
    cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    logger.info("Confusion Matrix (Rows=True, Cols=Pred, Labels=[0,1,2,3,4]):")
    logger.info(f"\n{cm_full}")
    
    # Check prediction distribution
    pred_counts = {i: y_pred.count(i) for i in range(5)}
    logger.info(f"Prediction Distribution: {pred_counts}")
    
    # For visualization, we still use the restricted labels [1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual (Ground Truth)')
    plt.title(f'Temporal Confusion Matrix (Acc={accuracy:.1f}%)')
    plt.savefig("temporal_confusion_matrix.png")
    
    # 5. HTML Report
    generate_html_report(accuracy, len(valid_samples), latent_dim)

def generate_html_report(accuracy, n_samples, latent_dim):
    def img_to_base64(path):
        if not os.path.exists(path): return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
            
    img_cm = img_to_base64("temporal_confusion_matrix.png")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Temporal Consistency Analysis</title>
        <style>
             body {{ font-family: sans-serif; margin: 20px; background: #f8f9fa; }}
            .container {{ max-width: 800px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .metric {{ font-size: 2em; font-weight: bold; color: #27ae60; margin: 10px 0; }}
            .details {{ color: #7f8c8d; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Temporal Consistency Analysis</h1>
            <div class="metric">{accuracy:.2f}% Accuracy</div>
            <p class="details">Evaluated on {n_samples} homogeneous trajectory segments (Actions strictly Left or Right).</p>
            <p class="details">Latent Dimension: {latent_dim}</p>
            
            <h3>Confusion Matrix</h3>
            <p>Checks if Start->Goal implies the correct Action in hindsight.</p>
            <img src="data:image/png;base64,{img_cm}" />
            
            <p style="margin-top: 30px; font-size: 0.9em; color: #999;">Generated by Embrained-AI</p>
        </div>
    </body>
    </html>
    """
    
    with open("temporal_consistency_report.html", "w") as f:
        f.write(html)
    logger.info("Saved temporal_consistency_report.html")
    print(f"Report generated: {os.path.abspath('temporal_consistency_report.html')}")

if __name__ == "__main__":
    main()
