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
import time
import json
import logging
import argparse
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import IMG_H, IMG_W, DATA_DIR, ACTION_PWM_MAP
from modules.spatial_model import ContrastiveVisuomotorEncoder, TinyVAE
from backend.train_vae import loss_function
from backend.train_vae import export_global_latents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainCVE")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = False  # Prevent CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH under AMP

# Map macro_action IDs to training indices (exclude STOP=0 and INTENTIONAL_STOP=5)
TRAINING_ACTION_MAP = {1: 0, 2: 1, 3: 2, 4: 3}  # FWD=0, REV=1, LEFT=2, RIGHT=3
NUM_TRAINING_ACTIONS = 4


class CVETransitionDataset(Dataset):
    """Dataset that loads consecutive (frame_t, action_t, frame_t+1) transition pairs.
    
    Builds temporal pairs from all_transitions.json, grouping by session to ensure
    consecutive frames are truly sequential. Filters out STOP actions.
    """
    def __init__(self, data_root, positive_window=3, negative_min_gap=20):
        self.data_root = data_root
        self.positive_window = positive_window
        self.negative_min_gap = negative_min_gap
        self.pairs = []  # List of (img_path_t, action_idx, img_path_tp1)
        self.all_image_paths = []  # For negative sampling
        
        json_path = os.path.join(data_root, "all_transitions.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find curated dataset at {json_path}")
            
        with open(json_path, "r") as f:
            all_data = json.load(f)
        
        # Group transitions by session
        sessions = {}
        for item in all_data:
            p = item.get('image_path', '')
            if 'webcam_frame_' in p or 'frame_' not in os.path.basename(p):
                continue
            
            abs_path = p if os.path.isabs(p) else os.path.join(self.data_root, p)
            if not os.path.exists(abs_path):
                continue
                
            session = item.get('session', 'default')
            if session not in sessions:
                sessions[session] = []
            
            macro_action = item.get('macro_action', 0)
            sessions[session].append({
                'abs_path': abs_path,
                'macro_action': macro_action,
                'timestamp': item.get('timestamp', 0)
            })
        
        # Build transition pairs within each session
        for session_name, frames in sessions.items():
            # Sort by timestamp to ensure temporal ordering
            frames.sort(key=lambda x: x['timestamp'])
            
            for i in range(len(frames) - 1):
                action_id = frames[i]['macro_action']
                
                # Skip STOP actions (0 and 5)
                if action_id not in TRAINING_ACTION_MAP:
                    continue
                
                training_action = TRAINING_ACTION_MAP[action_id]
                
                # Positive pair: current frame -> next frame (within window)
                for k in range(1, min(self.positive_window + 1, len(frames) - i)):
                    self.pairs.append((
                        frames[i]['abs_path'],
                        training_action,
                        frames[i + k]['abs_path']
                    ))
                    
                self.all_image_paths.append(frames[i]['abs_path'])
            
            # Add last frame to image pool
            if frames:
                self.all_image_paths.append(frames[-1]['abs_path'])
        
        # Standard augmentation for anchor (strong color jitter to filter TV/lighting)
        self.augment = transforms.Compose([
            # No Resize needed, images are cached resized
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
        
        # Clean transform (no augmentation)
        self.clean_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        logger.info(f"CVE Dataset: {len(self.pairs)} transition pairs from {len(sessions)} sessions")
        logger.info(f"Image pool: {len(self.all_image_paths)} frames for negative sampling")
        
        # Preload all unique images into RAM
        self.image_cache = {}
        unique_paths = list(set(self.all_image_paths))
        logger.info(f"Preloading {len(unique_paths)} unique frames into RAM to bypass disk I/O...")
        
        for i, p in enumerate(unique_paths):
            if i % 1000 == 0 and i > 0:
                logger.info(f"  Loaded {i}/{len(unique_paths)} frames...")
            try:
                img = Image.open(p).convert('RGB').resize((IMG_W, IMG_H))
                self.image_cache[p] = img
            except Exception:
                self.image_cache[p] = Image.new('RGB', (IMG_W, IMG_H))
                
        logger.info("Dataset preloading complete!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path_t, action_idx, img_path_tp1 = self.pairs[idx]
        
        img_t = self.image_cache.get(img_path_t)
        img_tp1 = self.image_cache.get(img_path_tp1)
        
        if img_t is None or img_tp1 is None:
            # Fallback
            return (
                torch.zeros((3, IMG_H, IMG_W)),
                torch.zeros((3, IMG_H, IMG_W)),
                torch.zeros((3, IMG_H, IMG_W)),
                torch.tensor(0, dtype=torch.long),
                torch.zeros((3, IMG_H, IMG_W))
            )
        
        # Anchor: clean version of frame_t
        anchor = self.clean_transform(img_t)
        
        # Positive: clean version of frame_t+1
        positive = self.clean_transform(img_tp1)
        
        # Augmented anchor: strongly augmented version of frame_t (for invariance loss)
        anchor_aug = self.augment(img_t)
        
        # Hard negative: random frame from distant part of dataset
        # False Negative Rejection: ensure negative is visually distinct from anchor
        MAX_RETRIES = 5
        for _ in range(MAX_RETRIES):
            neg_idx = random.randint(0, len(self.all_image_paths) - 1)
            neg_path = self.all_image_paths[neg_idx]
            neg_img = self.image_cache.get(neg_path, img_t)
            
            # Check visual similarity to prevent pushing visually identical states apart
            anchor_tensor = self.clean_transform(img_t)
            neg_tensor = self.clean_transform(neg_img)
            mse = torch.mean((anchor_tensor - neg_tensor)**2)
            if mse > 0.025: # Tunable threshold for "different enough"
                break
                
        negative = neg_tensor
        
        action_tensor = torch.tensor(action_idx, dtype=torch.long)
        
        return anchor, positive, anchor_aug, action_tensor, negative


def info_nce_loss(anchor_z, positive_z, negative_zs, temperature=0.07):
    """Compute InfoNCE contrastive loss.
    
    Args:
        anchor_z: (B, D) anchor embeddings
        positive_z: (B, D) positive embeddings  
        negative_zs: (B, D) negative embeddings (in-batch negatives also used)
        temperature: softmax temperature
    """
    # L2 normalize for cosine similarity
    anchor_norm = F.normalize(anchor_z, dim=-1)
    positive_norm = F.normalize(positive_z, dim=-1)
    negative_norm = F.normalize(negative_zs, dim=-1)
    
    # Positive similarity: (B,)
    pos_sim = torch.sum(anchor_norm * positive_norm, dim=-1, keepdim=True) / temperature
    
    # In-batch negatives: use all other samples in the batch as negatives
    # (B, B) similarity matrix
    all_sim = torch.mm(anchor_norm, positive_norm.t()) / temperature
    
    # Explicit negative similarity: (B, 1)
    neg_sim = torch.sum(anchor_norm * negative_norm, dim=-1, keepdim=True) / temperature
    
    # Concatenate: positive (first column) + in-batch negatives + explicit negative
    # For each anchor i, the positive is at position i in positive_norm
    # Use cross-entropy where the correct label is the diagonal
    labels = torch.arange(anchor_z.size(0), device=anchor_z.device)
    
    # Add explicit negatives as extra columns
    logits = torch.cat([all_sim, neg_sim], dim=1)  # (B, B+1)
    
    loss = F.cross_entropy(logits, labels)
    return loss


def train_cve(data_root, num_epochs=100, batch_size=256, lr=3e-4, latent_dim=32,
              temperature=0.07, model_size='large', stop_event=None, 
              progress_callback=None, model_filename=None):
    """Train a Contrastive Visuomotor Encoder."""
    
    logger.info(f"Starting CVE Training on {DEVICE}")
    logger.info(f"Latent Dim: {latent_dim}, Temperature: {temperature}, Model Size: {model_size}")
    
    dataset = CVETransitionDataset(data_root)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    logger.info(f"Dataset Size: {len(dataset)} transition pairs")
    
    model = ContrastiveVisuomotorEncoder(
        latent_dim=latent_dim,
        model_size=model_size,
        input_spatial_dim=IMG_W,
        n_actions=NUM_TRAINING_ACTIONS
    ).to(DEVICE)
    
    # [NEW] Dual-Train VAE for visual manifold/thresholding
    vae_model = TinyVAE(
        model_size=model_size, 
        latent_dim=latent_dim, 
        input_spatial_dim=IMG_W, 
        in_channels=3
    ).to(DEVICE)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=lr)
    vae_model.train()

    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    model.train()
    last_report_time = 0
    
    for epoch in range(num_epochs):
        if stop_event and stop_event.is_set():
            logger.info(f"Training interrupted by user at epoch {epoch}.")
            break
            
        total_contrastive = 0
        total_action = 0
        total_invariance = 0
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (anchor, positive, anchor_aug, actions, negative) in enumerate(dataloader):
            if stop_event and stop_event.is_set():
                break
                
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            anchor_aug = anchor_aug.to(DEVICE)
            actions = actions.to(DEVICE)
            negative = negative.to(DEVICE)
            
            optimizer.zero_grad()
            vae_optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    # Encode all frames
                    z_anchor = model.encode(anchor)
                    z_positive = model.encode(positive)
                    z_anchor_aug = model.encode(anchor_aug)
                    z_negative = model.encode(negative)
                    
                    # 1. InfoNCE contrastive loss
                    z_anchor_proj = model.project(z_anchor)
                    z_positive_proj = model.project(z_positive)
                    z_negative_proj = model.project(z_negative)
                    contrastive_loss = info_nce_loss(z_anchor_proj, z_positive_proj, z_negative_proj, temperature)
                    
                    # 2. Action prediction loss
                    action_logits = model.predict_action(z_anchor, z_positive)
                    action_loss = F.cross_entropy(action_logits, actions)
                    
                    # 3. Augmentation invariance loss
                    invariance_loss = F.mse_loss(z_anchor, z_anchor_aug)
                    
                    # Combined CVE loss
                    loss = contrastive_loss + 0.5 * action_loss + 0.3 * invariance_loss
                    
                    # [NEW] VAE Loss
                    current_fractional_epoch = epoch + (batch_idx + 1) / len(dataloader)
                    beta_warmup = min(1.0, current_fractional_epoch / 5.0) * 0.5
                    recon, mu, logvar = vae_model(anchor)
                    vae_loss, mse, kld = loss_function(recon, anchor, mu, logvar, beta=beta_warmup)
                
                scaler.scale(loss).backward()
                scaler.scale(vae_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.step(vae_optimizer)
                scaler.update()
            else:
                z_anchor = model.encode(anchor)
                z_positive = model.encode(positive)
                z_anchor_aug = model.encode(anchor_aug)
                z_negative = model.encode(negative)
                
                z_anchor_proj = model.project(z_anchor)
                z_positive_proj = model.project(z_positive)
                z_negative_proj = model.project(z_negative)
                contrastive_loss = info_nce_loss(z_anchor_proj, z_positive_proj, z_negative_proj, temperature)
                
                action_logits = model.predict_action(z_anchor, z_positive)
                action_loss = F.cross_entropy(action_logits, actions)
                
                invariance_loss = F.mse_loss(z_anchor, z_anchor_aug)
                
                loss = contrastive_loss + 0.5 * action_loss + 0.3 * invariance_loss
                
                # [NEW] VAE Loss
                current_fractional_epoch = epoch + (batch_idx + 1) / len(dataloader)
                beta_warmup = min(1.0, current_fractional_epoch / 5.0) * 0.5
                recon, mu, logvar = vae_model(anchor)
                vae_loss, mse, kld = loss_function(recon, anchor, mu, logvar, beta=beta_warmup)
                
                loss.backward()
                vae_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=5.0)
                optimizer.step()
                vae_optimizer.step()
            
            total_contrastive += contrastive_loss.item()
            total_action += action_loss.item()
            total_invariance += invariance_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            # Progress reporting
            current_time = time.time()
            current_fractional_epoch = epoch + (batch_idx + 1) / len(dataloader)
            if (current_time - last_report_time) >= 1.0 or batch_idx == 0:
                last_report_time = current_time
                if progress_callback:
                    avg_loss = total_loss / num_batches
                    avg_action = total_action / num_batches
                    progress_callback(
                        current_fractional_epoch, avg_loss,
                        kld=avg_action,  # Report action loss in KLD slot for UI chart
                        recon=total_contrastive / num_batches  # Contrastive in recon slot
                    )
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | NCE: {contrastive_loss.item():.4f} | "
                      f"Act: {action_loss.item():.4f} | Inv: {invariance_loss.item():.4f}", end='\r')
        
        if stop_event and stop_event.is_set():
            break
            
        scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_nce = total_contrastive / max(num_batches, 1)
        avg_act = total_action / max(num_batches, 1)
        avg_inv = total_invariance / max(num_batches, 1)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | NCE: {avg_nce:.4f} | "
                    f"Action: {avg_act:.4f} | Invariance: {avg_inv:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if progress_callback:
            progress_callback(epoch + 1, avg_loss, kld=avg_act, recon=avg_nce)
    
    # Always use canonical cve_ naming regardless of caller
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"cve_{latent_dim}d_{timestamp}"
    
    model_path = os.path.join(data_root, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"CVE Model saved to {model_path}")
    
    # [NEW] Save VAE
    vae_model_name = f"tinyvae-vae_{timestamp}.pth"
    vae_model_path = os.path.join(data_root, vae_model_name)
    torch.save(vae_model.state_dict(), vae_model_path)
    logger.info(f"Visual VAE Model saved to {vae_model_path}")
    
    # Export Global Latents for visualization and goal mapping
    logger.info("Extracting global structural latents...")
    export_global_latents(model, data_root, model_name, architecture='contrastive')
    logger.info("Global latent extraction complete.")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train Contrastive Visuomotor Encoder (CVE)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--model_size", type=str, default='medium', help="Encoder size")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory override")
    args = parser.parse_args()
    
    data_dir = args.data_dir or DATA_DIR
    
    train_cve(
        data_root=data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        temperature=args.temperature,
        model_size=args.model_size
    )


if __name__ == "__main__":
    main()
