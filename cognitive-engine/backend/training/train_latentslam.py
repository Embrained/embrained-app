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
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from .datasets.latentslam_dataset import LatentSLAMDataset
from ..models.latentslam import LatentSLAM
from backend.models.quantized_spatial import DiscreteLatentSLAM

def infonce_loss(features_q, features_k, temperature=0.1):
    """SimCLR-style InfoNCE matching positive pairs and repelling cross-batch negatives."""
    q = F.normalize(features_q, dim=1)
    k = F.normalize(features_k, dim=1)
    logits = torch.matmul(q, k.T) / temperature # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def kl_div_loss(mu, logvar, beta=1.0):
    """Standard VAE KLD Loss attached to N(0,I). clamped to prevent math explosion."""
    logvar = torch.clamp(logvar, min=-20, max=20)
    mu = torch.clamp(mu, min=-20, max=20)
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return beta * kld, kld

def train(data_root, num_epochs=100, stop_event=None, progress_callback=None, batch_size=64, learning_rate=1e-4, beta=2.0, transition_loss_weight=1.0, contrastive_weight=0.0, architecture='continuous', model_size='large', dataset_dirs=None, model_filename=None, image_size=64, num_layers=4, latent_dim=128, num_actions=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LatentSLAM on device: {device}")
    
    # 1. Setup Data
    trans_path = os.path.join(data_root, "all_transitions.json")
    if not os.path.exists(trans_path):
        print("ERROR: all_transitions.json not found. Please process a dataset first.")
        return None
        
    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    # (Delay dataloader initialization until after VAE geometry is inferred)
    
    # 2. Setup Base VAE & Infer Model Size
    import glob
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from modules.spatial_model import TinyVAE

    # Locate pre-trained Base VAE by scanning metadata files
    # We want a VAE matching the requested architecture (or continuous by default for older models)
    meta_files = glob.glob(os.path.join(data_root, "*-vae_*_meta.json")) + glob.glob(os.path.join(data_root, "models", "*-vae_*_meta.json"))
    meta_files = [m for m in meta_files if 'cql' not in m.lower()]
    
    valid_candidates = []
    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get("type") == "vae":
                hyperparams = meta.get("hyperparameters", {})
                file_arch = hyperparams.get("pipelineArchitecture", "continuous")
                if file_arch == architecture:
                    model_path = meta_path.replace("_meta.json", ".pth")
                    if os.path.exists(model_path):
                        valid_candidates.append(model_path)
        except Exception as e:
            print(f"Failed to read metadata {meta_path}: {e}")
            
    # Fallback to old tiny_vae_final.pth if we are dealing with 'continuous' and found nothing
    if not valid_candidates and architecture == 'continuous':
        old_candidates = [os.path.join(data_root, "tiny_vae_final.pth"), os.path.join(data_root, "models", "tiny_vae_final.pth")]
        valid_candidates = [c for c in old_candidates if os.path.exists(c)]

    if not valid_candidates:
        print(f"ERROR: Cannot train LatentSLAM. No pre-trained Base VAE found in data/ for architecture: {architecture}")
        return None
        
    # Sort by modification time to get the latest
    valid_candidates.sort(key=os.path.getmtime, reverse=True)
    vae_path = valid_candidates[0]
    
    base_state_dict = None
    
    try:
        print(f"Discovered Pre-trained Base VAE for [{architecture}]: {os.path.basename(vae_path)}")
        try:
            base_state_dict = torch.load(vae_path, map_location=device, weights_only=True)
        except Exception:
            base_state_dict = torch.load(vae_path, map_location=device)
            
        print("Introspecting base state dict mapping to force alignment...")
        from modules.spatial_model import TinyVAE
        det_latent_dim, det_model_size, det_img_dim, _ = TinyVAE.detect_size(base_state_dict)
        print(f"Introspection overrides: latent_dim={det_latent_dim}, size={det_model_size}, img={det_img_dim}")
        
        latent_dim = det_latent_dim
        model_size = det_model_size
        image_size = det_img_dim
        
    except Exception as e:
        print(f"Failed to load/parse VAE base weights: {e}")
        print("WARNING: Falling back to default geometry.")
        base_state_dict = None
            
    # Calculate hidden_dim dynamically based on model_size
    hidden_dim = 256
    model_size_lower = model_size.lower()
    if model_size_lower == "small": hidden_dim = 256
    elif model_size_lower == "medium": hidden_dim = 512
    elif model_size_lower == "large": hidden_dim = 1024
    elif model_size_lower == "enormous": hidden_dim = 2048
    elif model_size_lower == "tectonic": hidden_dim = 4096
    
    if architecture == "discrete":
        print(f"Initializing Discrete VQ-VAE model with hidden_dim={hidden_dim} (K=512, Size: {model_size}, Resol: {image_size}x{image_size})")
        model = DiscreteLatentSLAM(latent_dim=latent_dim, hidden_dim=hidden_dim, image_size=image_size, model_size=model_size, num_actions=num_actions, num_embeddings=512).to(device)
    else:
        print(f"Initializing Continuous LatentSLAM model with hidden_dim={hidden_dim} (Size: {model_size}, Resol: {image_size}x{image_size}, Actions: {num_actions})")
        model = LatentSLAM(latent_dim=latent_dim, hidden_dim=hidden_dim, image_size=image_size, model_size=model_size, num_actions=num_actions).to(device)
    
    # 2.4 Initialize Dataloader now that we know the correct image_size
    dataset = LatentSLAMDataset(all_data, data_root, device=device, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2.5 Load the Base VAE Weights into the newly matching LatentSLAM model
    if base_state_dict is not None:
        try:
            model.load_state_dict(base_state_dict, strict=False)
            print("Successfully loaded VAE spatial weights! Transition MLP will train from scratch.")
        except Exception as e:
            print(f"CRITICAL ERROR loading matched base weights: {e}")
            
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. Training Loop
    model.train()
    best_loss = float('inf')
    
    # Beta previously hardcoded to 2.0, now parameterized
    
    
    for epoch in range(num_epochs):
        if stop_event and stop_event.is_set():
            break
            
        epoch_loss = 0
        epoch_kl = 0
        epoch_recon = 0
        
        last_report_time = time.time()
        num_batches = len(dataloader)
        
        for i, (img_curr, action, img_next) in enumerate(dataloader):
            if stop_event and stop_event.is_set():
                break
                
            img_curr = img_curr.to(device)
            action = action.to(device)
            img_next = img_next.to(device)
            
            optimizer.zero_grad()
            
            # --- Core Evaluation passes logic explicitly split by architecture ---
            if architecture == "discrete":
                # Phase A: Discrete Spatial Component
                recon_curr, ze_curr, zq_curr, vq_loss_curr, perp_curr, idx_curr = model(img_curr)
                recon_next, ze_next, zq_next, vq_loss_next, perp_next, idx_next = model(img_next)
                
                recon_loss_curr = F.mse_loss(recon_curr, img_curr, reduction='none').sum(dim=[1, 2, 3]).mean()
                recon_loss_next = F.mse_loss(recon_next, img_next, reduction='none').sum(dim=[1, 2, 3]).mean()
                total_recon = recon_loss_curr + recon_loss_next
                
                # Replace KL Divergence with VQ Cost
                total_kl = vq_loss_curr + vq_loss_next
                raw_kl_curr = vq_loss_curr
                raw_kl_next = vq_loss_next
                
                # Phase B: Categorical Prediction MLP
                logits_all = model.predict_next_state(zq_curr.detach()) 
                batch_indices = torch.arange(img_curr.size(0), device=device)
                logits_next = logits_all[batch_indices, action] # [batch, K]
                
                # Transition Loss evaluates whether we snap to the correct ID!
                transition_loss = F.cross_entropy(logits_next, idx_next.detach())
                
                # Provide zq bounds for InfoNCE mapping downstream
                mu_curr = zq_curr
                mu_next = zq_next
            else:
                # Phase A: Continuous Spatial VAE
                recon_curr, mu_curr, logvar_curr = model(img_curr)
                recon_next, mu_next, logvar_next = model(img_next)
                
                recon_loss_curr = F.mse_loss(recon_curr, img_curr, reduction='none').sum(dim=[1, 2, 3]).mean()
                recon_loss_next = F.mse_loss(recon_next, img_next, reduction='none').sum(dim=[1, 2, 3]).mean()
                total_recon = recon_loss_curr + recon_loss_next
                
                kl_loss_curr, raw_kl_curr = kl_div_loss(mu_curr, logvar_curr, beta)
                kl_loss_next, raw_kl_next = kl_div_loss(mu_next, logvar_next, beta)
                total_kl = kl_loss_curr + kl_loss_next
                
                # Phase B: Vector Transition MLP
                hat_mu_all = model.predict_next_state(mu_curr.detach()) 
                batch_indices = torch.arange(img_curr.size(0), device=device)
                hat_mu_next = hat_mu_all[batch_indices, action] 
                
                transition_loss = F.mse_loss(hat_mu_next, mu_next.detach(), reduction='none').sum(dim=-1).mean()
                
            # --- Phase C: InfoNCE Disentanglement (Topological Push) ---
            c_loss = 0.0
            if contrastive_weight > 0:
                c_loss = infonce_loss(mu_curr, mu_next.detach())

            # --- Combine Losses ---
            loss = (total_recon + total_kl) * 0.5 + (transition_loss * transition_loss_weight) + (c_loss * contrastive_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_kl += ((raw_kl_curr + raw_kl_next) * 0.5).item()
            epoch_recon += (total_recon * 0.5).item()

            # Fine-grained Progress Updates (Time-based ~1Hz)
            current_time = time.time()
            if (current_time - last_report_time) >= 1.0 or i == 0 or (i + 1) == num_batches:
                last_report_time = current_time
                avg_loss_running = epoch_loss / (i + 1)
                avg_kl_running = epoch_kl / (i + 1)
                avg_recon_running = epoch_recon / (i + 1)
                current_fractional_epoch = epoch + (i + 1) / num_batches
                
                if progress_callback:
                    progress_callback(current_fractional_epoch, avg_loss_running, avg_kl_running, avg_recon_running)
            
        avg_loss = epoch_loss / len(dataloader)
        avg_kl = epoch_kl / len(dataloader)
        avg_recon = epoch_recon / len(dataloader)
        
        # Already reported final for epoch in loop, but redundancy at epoch boundary is fine
        if progress_callback:
            progress_callback(epoch + 1, avg_loss, avg_kl, avg_recon)
            
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} (KL: {avg_kl:.4f}, Recon: {avg_recon:.4f})")
        
        # Save check
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save Weights
            save_path = os.path.join(data_root, "latentslam_best.pth")
            torch.save(model.state_dict(), save_path)
            
    # Save Final with Timestamp or Explicit Name
    if model_filename:
        # Ensure it ends with .pth
        if not model_filename.endswith(".pth"):
            model_filename += ".pth"
        filename = model_filename
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"latentslam_{timestamp}.pth"
        
    final_path = os.path.join(data_root, filename)
    torch.save(model.state_dict(), final_path)
    
    return final_path


