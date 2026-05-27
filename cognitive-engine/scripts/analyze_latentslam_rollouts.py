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
import logging
import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LatentSLAM-Eval")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.latentslam import LatentSLAM

DATASET_NAME = "nook"
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_PATH = os.path.join(DATA_ROOT, "latentslam_20260308_172729.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 64 # Default, updated after loading model

def get_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

def load_image(path, img_size):
    img = cv2.imread(path)
    if img is None:
        return torch.zeros((1, 3, img_size, img_size)).to(DEVICE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    return get_transform(img_size)(img_pil).unsqueeze(0).to(DEVICE)

def discretize_action(left, right):
    tol = 40
    if abs(left) < 1 and abs(right) < 1: return 0  # STOP
    if left < -tol and right > tol: return 1  # FWD
    if left < -tol and right < -tol: return 3  # LEFT
    if left > tol and right > tol: return 4  # RIGHT
    if left > tol and right < -tol: return 2  # BACK (Deprecated, but mapped just in case)
    return 0

def action_id_to_tensor(act_id):
    # Matches engine.py ACTION_PWM_MAP
    # {0: (0,0), 1: (-255, 255), 2: (255, -255), 3: (-255, -255), 4: (255, 255)}
    if act_id == 1: pwm = [-255, 255]
    elif act_id == 2: pwm = [255, -255]
    elif act_id == 3: pwm = [-255, -255]
    elif act_id == 4: pwm = [255, 255]
    else: pwm = [0, 0]
    return torch.tensor([pwm], dtype=torch.float32).to(DEVICE) / 255.0

def load_latentslam():
    logger.info(f"Loading LatentSLAM from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    
    hidden_dim = 256
    latent_dim = 128
    num_layers = 4
    image_size = IMG_SIZE
    
    if 'posterior_fc.0.weight' in state_dict:
        hidden_dim = state_dict['posterior_fc.0.weight'].shape[0]
        latent_dim = state_dict['posterior_fc.2.weight'].shape[0] // 2
        
        if 'encoder.8.weight' in state_dict: 
            num_layers = 5
            
        flattened_size = state_dict['posterior_fc.0.weight'].shape[1] - latent_dim - 2
        import math
        if num_layers == 5:
            image_size = int(math.sqrt(flattened_size / 512) * 32)
        else:
            image_size = int(math.sqrt(flattened_size / 256) * 16)

    logger.info(f"Detected Arch: Latent={latent_dim}, Hidden={hidden_dim}, Layers={num_layers}, ImgSize={image_size}")
    
    model = LatentSLAM(
        latent_dim=latent_dim, 
        hidden_dim=hidden_dim, 
        image_size=image_size, 
        num_layers=num_layers
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def analyze_1step_transitions():
    model = load_latentslam()
    transitions_path = os.path.join(DATA_ROOT, 'all_transitions.json')
    with open(transitions_path, 'r') as f:
        all_data = json.load(f)

    logger.info("Extracting 1000 random forward 1-step transitions...")
    fwd_transitions = []
    
    for i in range(len(all_data) - 1):
        step_t = all_data[i]
        step_t1 = all_data[i+1]
        
        # Must be same session
        if step_t['session'] != step_t1['session']: continue
        
        l = step_t.get('left_cmd', 0)
        r = step_t.get('right_cmd', 0)
        act_id = discretize_action(l, r)
        
        if act_id == 1: # FWD
            fwd_transitions.append((step_t['image_path'], step_t1['image_path']))
            
    random.shuffle(fwd_transitions)
    eval_fwd = fwd_transitions[:1000]
    
    logger.info(f"Evaluating {len(eval_fwd)} forward transitions...")
    
    mses, cosines, l1_losses = [], [], []
    recon_ssims, recon_psnrs = [], []
    pred_ssims, pred_psnrs = [], []
    
    with torch.no_grad():
        for start_path, next_path in eval_fwd:
            img_s = load_image(os.path.join(DATA_ROOT, start_path), model.image_size)
            img_n = load_image(os.path.join(DATA_ROOT, next_path), model.image_size)
            
            zero_state = torch.zeros(1, model.latent_dim).to(DEVICE)
            zero_act = torch.zeros(1, 2).to(DEVICE)
            
            # 1. State at T
            mu_s, _ = model.get_posterior(zero_state, zero_act, img_s)
            
            # 2. State at T+1 (Ground Truth via posterior)
            mu_n_gt, _ = model.get_posterior(zero_state, zero_act, img_n)
            
            # 3. State at T+1 (Predicted via prior + fwd action)
            act_fwd = action_id_to_tensor(1)
            mu_n_pred, _ = model.get_prior(mu_s, act_fwd)
            
            # Latent Metrics
            mse = torch.nn.functional.mse_loss(mu_n_pred, mu_n_gt).item()
            cos = torch.nn.functional.cosine_similarity(mu_n_pred, mu_n_gt).item()
            mses.append(mse)
            cosines.append(cos)
            
            # Image Metrics (Original vs Autoencoder vs Hallucination)
            gt_img_np = img_n[0].cpu().permute(1, 2, 0).numpy()
            recon_gt = model.get_likelihood(mu_n_gt)[0].cpu().permute(1, 2, 0).numpy()
            recon_pred = model.get_likelihood(mu_n_pred)[0].cpu().permute(1, 2, 0).numpy()
            
            l1_loss = np.mean(np.abs(gt_img_np - recon_pred))
            l1_losses.append(l1_loss)
            
            # SSIM & PSNR calculation
            # Calculate structural retention of the Autoencoder baseline
            val_ssim_gt = ssim(gt_img_np, recon_gt, data_range=1.0, channel_axis=2)
            val_psnr_gt = psnr(gt_img_np, recon_gt, data_range=1.0)
            recon_ssims.append(val_ssim_gt)
            recon_psnrs.append(val_psnr_gt)
            
            # Calculate structural loss of the Hallucination relative to baseline
            val_ssim_pred = ssim(gt_img_np, recon_pred, data_range=1.0, channel_axis=2)
            val_psnr_pred = psnr(gt_img_np, recon_pred, data_range=1.0)
            pred_ssims.append(val_ssim_pred)
            pred_psnrs.append(val_psnr_pred)
            
    logger.info(f"--- 1-Step Transition Results (N={len(eval_fwd)}) ---")
    logger.info(f"Latent Cosine Sim:    {np.mean(cosines):.4f} +/- {np.std(cosines):.4f}")
    logger.info(f"Latent MSE:           {np.mean(mses):.4f} +/- {np.std(mses):.4f}")
    logger.info(f"Image Pred L1 Loss:   {np.mean(l1_losses):.4f} +/- {np.std(l1_losses):.4f}")
    logger.info(f"Autoencoder SSIM:     {np.mean(recon_ssims):.4f} +/- {np.std(recon_ssims):.4f}")
    logger.info(f"Hallucination SSIM:   {np.mean(pred_ssims):.4f} +/- {np.std(pred_ssims):.4f}")
    logger.info(f"Autoencoder PSNR:     {np.mean(recon_psnrs):.4f} +/- {np.std(recon_psnrs):.4f}")
    logger.info(f"Hallucination PSNR:   {np.mean(pred_psnrs):.4f} +/- {np.std(pred_psnrs):.4f}")
    
    return model, eval_fwd

def plot_hallucinations(model, eval_fwd):
    logger.info("Generating 5 Hallucinated 1-Step Forward Sequences...")
    
    fig, axes = plt.subplots(5, 2, figsize=(8, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    zero_state = torch.zeros(1, model.latent_dim).to(DEVICE)
    zero_act = torch.zeros(1, 2).to(DEVICE)
    act_fwd = action_id_to_tensor(1)
    
    with torch.no_grad():
        for i in range(5):
            start_path, next_path = eval_fwd[i]
            img_s = load_image(os.path.join(DATA_ROOT, start_path), model.image_size)
            
            # Original Image
            ax = axes[i, 0]
            display_img = img_s[0].cpu().permute(1, 2, 0).numpy()
            ax.imshow(display_img)
            ax.set_title("Original Frame")
            ax.axis('off')
            
            # Encode Base State
            curr_state, _ = model.get_posterior(zero_state, zero_act, img_s)
            
            # Rollout 1 step forward
            for step in range(1, 2):
                # Predict next via Prior
                curr_state, _ = model.get_prior(curr_state, act_fwd)
                
                # Decode hallucination
                recon = model.get_likelihood(curr_state)
                display_recon = recon[0].cpu().permute(1, 2, 0).numpy()
                
                ax = axes[i, step]
                ax.imshow(display_recon)
                ax.set_title(f"Hallucination +{step}")
                ax.axis('off')
                
    plt.savefig(os.path.join(DATA_ROOT, "latentslam_hallucinations.png"))
    logger.info(f"Saved hallucinations to {os.path.join(DATA_ROOT, 'latentslam_hallucinations.png')}")

if __name__ == "__main__":
    model, eval_fwd = analyze_1step_transitions()
    plot_hallucinations(model, eval_fwd)
