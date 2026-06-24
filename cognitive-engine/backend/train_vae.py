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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import time
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import json
import time
from PIL import Image

# Helper
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_H, IMG_W, DATA_DIR, MODELS_DIR
from modules.spatial_model import TinyVAE
from backend.models.quantized_spatial import DiscreteLatentSLAM

# ---------------------------------------------------------
# GLOBAL TRAINING CONFIGURATION
# ---------------------------------------------------------
EPOCHS = 50
BETA_WARMUP_EPOCHS = 5 # Graceful warmup to prevent initial shock collapse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainVAE")


class VAEDataset(Dataset):
    def __init__(self, data_root, selected_datasets=None):
        from backend.services.datasets import DatasetService
        self.ds_service = DatasetService(data_root)
        self.samples = []
        
        # If no selected_datasets, assume we want all folders in data_root
        if not selected_datasets:
            datasets_info = self.ds_service.list_datasets(fast=True)
            selected_datasets = [d['name'] for d in datasets_info['datasets']]
            
        for ds_name in selected_datasets:
            # Check for exact directory first
            exact_path = os.path.join(data_root, ds_name)
            if os.path.isdir(exact_path):
                matching_dirs = [exact_path]
            else:
                import re
                # Pattern ensures we only match the exact prefix, avoiding pool crossover (e.g. markov vs markov_simulation)
                pattern = re.compile(rf'^{re.escape(ds_name)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})$')
                matching_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) 
                               if os.path.isdir(os.path.join(data_root, d)) and pattern.match(d)]
            
            for ds_path in matching_dirs:
                if not os.path.exists(os.path.join(ds_path, "episode_data.csv")):
                    continue
                
                logger.debug(f"VAE Dataset: Loading from {ds_path}...")
                transitions = self.ds_service.load_transitions(ds_path)
                for t in transitions:
                    if t['format'] == 'markov':
                        t['image_path'] = os.path.join(ds_path, t['image_path'])
                    
                    self.samples.append(t)

        if not self.samples:
            logger.warning(f"VAE Dataset: No images found in {selected_datasets}.")

        logger.debug(f"VAE Dataset: Final count = {len(self.samples)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(), # [0, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            if 'image_path' in sample and os.path.exists(sample['image_path']):
                # Standard egocentric loading only
                img = Image.open(sample['image_path']).convert('RGB')
                return self.transform(img) # [3, H, W]
            else:
                raise ValueError("No image source found")
        except Exception as e:
            # Fallback
            return torch.zeros((3, IMG_H, IMG_W))

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # MSE Loss
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + beta * KLD, MSE, KLD


def train(data_root=DATA_DIR, num_epochs=20, stop_event=None, progress_callback=None, batch_size=128, learning_rate=1e-5, beta=0.5, model_size='large', selected_datasets=None, model_filename=None, architecture='continuous', latent_dim=128):      
    
    logger.info(f"Input Resolution: {IMG_W}x{IMG_H}")
    logger.info(f"Hyperparameters: Batch Size={batch_size}, LR={learning_rate}, Size={model_size}, Topology={architecture}, Latent={latent_dim}, BETA={beta}, Warmup_Epochs={BETA_WARMUP_EPOCHS}")
    
    dataset = VAEDataset(data_root, selected_datasets=selected_datasets)
    if len(dataset) == 0:
        logger.error("No data found!")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Auto-detect channels from dataset
    detect_sample = dataset[0]
    in_channels = detect_sample.shape[0] if isinstance(detect_sample, torch.Tensor) else 3
    logger.info(f"Auto-Detected Input Channels: {in_channels}")
    
    if architecture == 'discrete':
        logger.info(f"Initializing DiscreteVQVAE Vision Base... (Codebook: 128x{latent_dim})")
        model = DiscreteLatentSLAM(latent_dim=latent_dim, num_actions=1, model_size=model_size, image_size=IMG_W).to(DEVICE)
    else:
        logger.info("Initializing Continuous bVAE Vision Base...")
        model = TinyVAE(model_size=model_size, latent_dim=latent_dim, input_spatial_dim=IMG_W, in_channels=in_channels).to(DEVICE)
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    model.train()
    
    last_report_time = 0

    for epoch in range(num_epochs):
        # Stop Check
        if stop_event and stop_event.is_set():
            logger.info(f"Training interrupted by user at epoch {epoch}.")
            break
            
        total_loss = 0
        total_mse = 0
        total_kld = 0
        total_perplexity = 0
        perplexity_count = 0
        samples_processed = 0
        
        # Use simple loop if we have progress callback, or tqdm if not
        # To avoid tqdm interfering with logs if we are effectively running in non-interactive
        if progress_callback is None:
             iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
             iterator = dataloader

        batch_idx = 0
        num_batches = len(dataloader)

        for imgs in iterator:
            # Intra-batch stop check
            if stop_event and stop_event.is_set():
                break

            current_fractional_epoch = epoch + (batch_idx + 1) / num_batches
            
            imgs = imgs.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Calculate dynamic beta for KL annealing
            current_beta = min(1.0, current_fractional_epoch / BETA_WARMUP_EPOCHS) * beta
            
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                if architecture == 'discrete':
                    recon, z_e, z_q, vq_loss, perplexity, indices = model(imgs)
                    
                    # 1. OPTIMIZER: Use strictly balanced mean-reduction for stable AMP gradients
                    mse_mean = nn.functional.mse_loss(recon, imgs, reduction='mean')
                    
                    # 2. Entropy bonus: penalize low codebook utilization
                    code_counts = torch.bincount(indices.view(-1), minlength=model.num_embeddings).float()
                    code_probs = code_counts / code_counts.sum().clamp(min=1)
                    entropy = -torch.sum(code_probs * torch.log(code_probs + 1e-10))
                    max_entropy = torch.log(torch.tensor(float(model.num_embeddings), device=imgs.device))
                    entropy_loss = 0.1 * (max_entropy - entropy) / max_entropy  # Normalized [0, 0.1]
                    
                    loss_backward = mse_mean + vq_loss + entropy_loss
                    
                    # 3. LOGGING: Re-scale up to match the intrinsic sum-reduction logging of Continuous bVAE!
                    scale_factor = imgs.size(0) * imgs.size(1) * imgs.size(2) * imgs.size(3)
                    mse = mse_mean * scale_factor
                    kld = vq_loss * scale_factor
                    loss = loss_backward * scale_factor
                    
                    # Track perplexity (effective codebook utilization)
                    total_perplexity += perplexity.item()
                    perplexity_count += 1
                else:
                    recon, mu, logvar = model(imgs)
                    loss, mse, kld = loss_function(recon, imgs, mu, logvar, beta=current_beta)
                    loss_backward = loss
            
            if scaler:
                scaler.scale(loss_backward).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_backward.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            
            loop_loss = loss.item()
            total_loss += loop_loss
            total_mse += mse.item()
            total_kld += kld.item()
            samples_processed += imgs.size(0)
            
            # [NEW] Fine-grained Progress Updates (Time-based ~1Hz)
            current_time = time.time()
            if (current_time - last_report_time) >= 1.0 or batch_idx == 0 or (batch_idx + 1) == num_batches:
                last_report_time = current_time
                avg_loss = total_loss / samples_processed
                avg_kld = total_kld / samples_processed
                avg_mse = total_mse / samples_processed
                
                # logger.debug(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{num_batches} | Loss: {avg_loss:.4f} | KLD: {avg_kld:.4f}")
                
                if progress_callback:
                    res = progress_callback(current_fractional_epoch, avg_loss, kld=avg_kld, recon=avg_mse)
                    if isinstance(res, dict) and res.get('request_checkpoint'):
                        checkpoint_path = os.path.join(data_root, "checkpoint_vae.pth")
                        torch.save(model.state_dict(), checkpoint_path)
                        # Signal completion
                        progress_callback(current_fractional_epoch, avg_loss, kld=avg_kld, recon=avg_mse, checkpoint_ready=checkpoint_path)
            
            batch_idx += 1

            if progress_callback is None:
                 iterator.set_postfix(loss=loop_loss/batch_size)
            
        if stop_event and stop_event.is_set():
            break

        avg_loss = total_loss / len(dataset)
        avg_mse = total_mse / len(dataset)
        avg_kld = total_kld / len(dataset)
        
        if architecture == 'discrete' and perplexity_count > 0:
            avg_perplexity = total_perplexity / perplexity_count
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | VQ: {avg_kld:.4f} | Perplexity: {avg_perplexity:.1f}/{model.num_embeddings}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | KLD: {avg_kld:.4f}")
        
        if progress_callback:
            progress_callback(epoch+1, avg_loss, kld=avg_kld, recon=avg_mse)
            
    # Save Final with Timestamp or Explicit Name
    if model_filename:
        # Ensure it ends with .pth
        if not model_filename.endswith(".pth"):
            model_filename += ".pth"
        model_name = model_filename.replace(".pth", "")
    else:
        parent_name = os.path.basename(os.path.normpath(data_root))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if not parent_name or parent_name == 'data' or parent_name == '.':
            model_name = f"tiny_vae_{timestamp}"
        else:
            model_name = f"{parent_name}-vae_{timestamp}"
        
    final_path = os.path.join(data_root, f"{model_name}.pth")
         
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved VAE to {final_path}")
    
    # [NEW] Unified Exhaustive Topological Extrapolation
    export_global_latents(model, data_root, model_name, architecture)
    
    return final_path

# [NEW] End-of-Pipeline Exhaustive Topological Caching
def export_global_latents(model, data_root, model_name, architecture='continuous'):
    """
    Exhaustively extracts representations for every valid topological trajectory frame.
    Generates a highly optimized dual-keyed PyTorch dictionary for downstream caching.
    """
    import os
    import json
    import torch
    from PIL import Image
    from torchvision import transforms
    from config import IMG_H, IMG_W
    
    logger.info("\nInitializing global topological latent extraction! (This will be verbose and take a few minutes...)")
    trans_path = os.path.join(data_root, "all_transitions.json")
    if not os.path.exists(trans_path):
        logger.warning(f"Could not find {trans_path}, skipping absolute mapping.")
        return
        
    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])
    
    latent_dict_path = {}
    latent_dict_ts = {}
    
    model.eval()
    count = 0
    total = len(all_data)
    
    with torch.no_grad():
        for i, node in enumerate(all_data):
            img_path = node.get('image_path', '')
            if 'frame_' in img_path:
                ts = os.path.basename(img_path).replace('frame_', '').replace('.jpg', '')
            else:
                ts = None
                
            abs_path = os.path.join(data_root, img_path)
            if os.path.exists(abs_path):
                try:
                    img = Image.open(abs_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                    if architecture == 'discrete':
                        if 'DiscreteVQVAE' in type(model).__name__:
                            _, z_e, _, _, _ = model(img_tensor)
                        else:
                            _, z_e, _, _, _, _ = model(img_tensor)
                        val = z_e.squeeze(0).cpu()
                    else:
                        _, mu, _ = model(img_tensor)
                        val = mu.squeeze(0).cpu()
                    latent_dict_path[img_path] = val
                    if ts:
                        latent_dict_ts[ts] = val
                    
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"   ... Globally Compiled {count}/{total} frames.")
                except Exception as e:
                    pass
                    
    cache_path = os.path.join(data_root, f"{model_name}_global_latents.pt")
    combined = {
        "path_map": latent_dict_path,
        "ts_map": latent_dict_ts
    }
    torch.save(combined, cache_path)
    logger.info(f"Compilation Complete! Saved {count} absolute topologies natively mapped into {cache_path}")

if __name__ == "__main__":
    train()
