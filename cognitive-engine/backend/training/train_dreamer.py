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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy
import random
import cv2
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import math
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HIDDEN_DIM, INPUT_DIM, MODELS_DIR, ACTION_DIM, ACTION_PWM_MAP

from modules.spatial_model import TinyVAE
from backend.models.dreamerv3 import DreamerPolicy

logger = logging.getLogger("TrainDreamerV3")
logging.basicConfig(level=logging.INFO)

# --- Hyperparameters ---
SEQ_LEN = 16 # Length of imagined trajectory chunks
GAMMA = 0.99 
IMG_H = 64
IMG_W = 64

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
])

class DreamerSequenceDataset(Dataset):
    def __init__(self, trajectories, data_root, device, seq_len=16):
        self.samples = []
        self.data_root = data_root
        self.device = device
        self.seq_len = seq_len
        self.video_caps = {}
        
        logger.debug("DreamerDataset: Extracting sequences...")
        
        for idx, traj in enumerate(trajectories):
            seqlen = len(traj)
            if seqlen < self.seq_len:
                continue
                
            # Create overlapping sequences
            step = max(1, self.seq_len // 4)
            for start_idx in range(0, seqlen - self.seq_len + 1, step):
                chunk = traj[start_idx : start_idx + self.seq_len]
                self.samples.append(chunk)

        random.shuffle(self.samples)
        logger.info(f"DreamerDataset: Created {len(self.samples)} sequence chunks of length {self.seq_len}.")
        
    def __del__(self):
        if hasattr(self, 'video_caps'):
            for path, obj in self.video_caps.items():
                if hasattr(obj, 'isOpened') and obj.isOpened():
                    obj.release()
            self.video_caps.clear()

    def _get_frame_from_video(self, video_path, frame_idx):
        if not os.path.exists(video_path): return None
        if video_path not in self.video_caps:
            MAX_RAM_VIDEOS = 200
            if len(self.video_caps) >= MAX_RAM_VIDEOS:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return cv2.resize(frame_rgb, (IMG_W, IMG_H))
                return None
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return None
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (IMG_W, IMG_H))
                frames.append(frame_resized)
            cap.release()
            self.video_caps[video_path] = frames
            
        frames = self.video_caps[video_path]
        if frame_idx < len(frames): return frames[frame_idx]
        return None

    def _load_img(self, node):
        if not node: return torch.zeros((3, IMG_H, IMG_W))
        try:
            if 'image_path' in node:
                p = node['image_path']
                if not os.path.isabs(p): p = os.path.join(self.data_root, p)
                if os.path.exists(p):
                    img = cv2.imread(p)
                    if img is not None:
                        return transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            pass
        return torch.zeros((3, IMG_H, IMG_W))

    def _extract_action(self, node):
        if 'macro_action' in node:
            act = int(node['macro_action'])
            if act >= ACTION_DIM: act = 0
            return act
        raw_l = float(node.get('left_cmd', 0.0))
        raw_r = float(node.get('right_cmd', 0.0))
        best_act = 0
        best_dist = float('inf')
        for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
            dist = math.hypot(raw_l - map_l, raw_r - map_r)
            if dist < best_dist:
                best_dist, best_act = dist, act_id
        return best_act
        
    def _extract_state(self, node):
        dist = float(node.get('sonar', 0.0))
        # Continuous reward function based on sonar to avoid obstacles
        r = 1.0 if dist > 20.0 else -1.0
        return r

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        chunk = self.samples[idx]
        imgs, acts, rews = [], [], []
        
        for node in chunk:
            imgs.append(self._load_img(node))
            acts.append(self._extract_action(node))
            rews.append(self._extract_state(node))
            
        return (
            torch.stack(imgs), # (L, 3, H, W)
            torch.tensor(acts, dtype=torch.long), # (L)
            torch.tensor(rews, dtype=torch.float) # (L)
        )

def train(data_root, num_epochs=20, stop_event=None, progress_callback=None, batch_size=32, learning_rate=1e-4, tag="red_ball", dataset_dirs=None, model_filename=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training DreamerV3 on {device}")

    trans_path = os.path.join(data_root, "all_transitions.json")
    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"all_transitions.json not found in {data_root}")

    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    sessions = {}
    for item in all_data:
        s = item['session']
        if s not in sessions: sessions[s] = []
        sessions[s].append(item)
        
    trajectories = [sorted(sessions[s], key=lambda x: x['timestamp']) for s in sessions if len(sessions[s]) >= SEQ_LEN]
    
    dataset = DreamerSequenceDataset(trajectories, data_root, device, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Needs a prior VAE or uses raw pixels? DreamerV3 typically trains an encoder end-to-end,
    # but to be efficient we use a pre-trained VAE from our pipeline.
    # For now, let's auto-discover the VAE like train_cql does.
    import glob
    candidates = glob.glob(os.path.join(data_root, "*-vae_*.pth")) + glob.glob(os.path.join(MODELS_DIR, "*-vae_*.pth"))
    candidates = [c for c in candidates if "cql" not in c.lower() and "dreamer" not in c.lower()]
    if not candidates:
        raise FileNotFoundError("Could not find a VAE model to extract latents for DreamerV3.")
    
    candidates.sort(key=os.path.getmtime, reverse=True)
    vae_path = candidates[0]
    
    # Load VAE to extract latent dimensions
    temp_state = torch.load(vae_path, map_location=device, weights_only=True)
    latent_dim_det, model_size_det, input_spatial_dim_det, _ = TinyVAE.detect_size(temp_state)
    logger.info(f"Loaded feature encoder: {vae_path} (Latent: {latent_dim_det}, Spatial: {input_spatial_dim_det})")
    
    encoder = TinyVAE(latent_dim=latent_dim_det, model_size=model_size_det, input_spatial_dim=input_spatial_dim_det).to(device)
    encoder.load_state_dict(temp_state)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    
    # Init Dreamer Policy
    # We use continuous one-hot actions embedding internally
    dreamer = DreamerPolicy(action_dim=ACTION_DIM, obs_dim=latent_dim_det, hidden_dim=256, state_dim=32).to(device)
    optimizer_wm = optim.Adam(dreamer.world_model.parameters(), lr=learning_rate)
    optimizer_actor = optim.Adam(dreamer.actor.parameters(), lr=learning_rate)
    optimizer_critic = optim.Adam(dreamer.critic.parameters(), lr=learning_rate)

    logger.info(f"Starting Training Loop for {num_epochs} epochs...")
    if progress_callback: progress_callback(0, 0)
    
    last_report_time = 0
    
    for epoch in range(num_epochs):
        if stop_event and stop_event.is_set(): break
        
        total_loss_wm = 0
        total_loss_actor = 0
        num_batches = len(dataloader)
        
        for i, (imgs, acts, rews) in enumerate(dataloader):
            if stop_event and stop_event.is_set(): break
            
            # Move to device and reshape
            # imgs: (B, L, C, H, W)
            B, L, C, H, W = imgs.shape
            imgs = imgs.to(device).view(B * L, C, H, W)
            acts = acts.to(device)
            rews = rews.to(device)
            
            with torch.no_grad():
                # Encode images to obs
                feat = encoder.encoder(imgs)
                z_mu = encoder.fc_mu(feat)
                obs_t = z_mu.view(B, L, -1) # (B, L, ObsDim)
                
            acts_onehot = F.one_hot(acts, num_classes=ACTION_DIM).float() # (B, L, ActionDim)
            
            # --- 1. Train World Model ---
            h_t = torch.zeros(B, dreamer.hidden_dim, device=device)
            z_t = torch.zeros(B, dreamer.state_dim, device=device) # Prior mean/std placeholder
            prev_a = torch.zeros(B, ACTION_DIM, device=device)
            
            wm_loss = 0
            
            for t in range(L):
                # Step RSSM
                h_t, z_t, prior, post = dreamer.world_model.rssm.step(h_t, z_t, prev_a, obs_t[:, t])
                
                # Predict obs and rewards from posterior
                obs_pred, rew_pred = dreamer.world_model.forward_state(h_t, z_t)
                
                # Compute losses
                loss_obs = F.mse_loss(obs_pred, obs_t[:, t])
                loss_rew = F.mse_loss(rew_pred.squeeze(-1), rews[:, t])
                
                # KL Loss between posterior and prior (Information Bottleneck)
                prior_mean, prior_std = prior
                post_mean, post_std = post
                prior_dist = torch.distributions.Normal(prior_mean, prior_std)
                post_dist = torch.distributions.Normal(post_mean, post_std)
                kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
                
                wm_loss += loss_obs + loss_rew + 0.1 * kl_loss
                prev_a = acts_onehot[:, t]
                
            wm_loss = wm_loss / L
            optimizer_wm.zero_grad()
            wm_loss.backward()
            optimizer_wm.step()
            
            total_loss_wm += wm_loss.item()
            
            # --- 2. Train Actor Critic via Imagination ---
            # (Simplified: Imagine 15 steps starting from the last state of the real trajectory chunk)
            h_t_imag = h_t.detach()
            z_t_imag = z_t.detach()
            
            actor_loss = 0
            critic_loss = 0
            IMAGINE_HORIZON = 15
            
            # Collect imagined trajectory
            imag_h = []
            imag_z = []
            imag_a = []
            imag_a_logprob = []
            
            for t in range(IMAGINE_HORIZON):
                # Sample action from actor
                mean_a, std_a = dreamer.actor(h_t_imag, z_t_imag)
                dist_a = torch.distributions.Normal(mean_a, std_a)
                a_t_imag = dist_a.rsample()
                log_prob_a = dist_a.log_prob(a_t_imag).sum(dim=-1)
                
                # Convert continuous action back to valid discrete domain vector for RSSM
                # (We just use the raw rsampled values for the recurrent input here)
                
                # Step RSSM (no real observation -> sample from prior)
                h_t_imag, z_t_imag, _, _ = dreamer.world_model.rssm.step(h_t_imag, z_t_imag, a_t_imag, obs=None)
                
                imag_h.append(h_t_imag)
                imag_z.append(z_t_imag)
                imag_a.append(a_t_imag)
                imag_a_logprob.append(log_prob_a)
                
            imag_h = torch.stack(imag_h, dim=1)
            imag_z = torch.stack(imag_z, dim=1)
            imag_a_logprob = torch.stack(imag_a_logprob, dim=1)
            
            # Compute imagined rewards and values
            _, rew_imag = dreamer.world_model.forward_state(imag_h, imag_z) # (B, H, 1)
            val_imag = dreamer.critic(imag_h.detach(), imag_z.detach()) # (B, H, 1)
            
            # Compute targets (Lambda Return)
            target_vals = torch.zeros_like(val_imag)
            last_val = 0
            for t in reversed(range(IMAGINE_HORIZON)):
                target_vals[:, t] = rew_imag[:, t] + GAMMA * last_val
                last_val = target_vals[:, t]
                
            # Actor loss: Maximize expected return
            act_loss = -(target_vals.squeeze(-1).detach() * imag_a_logprob).mean()
            
            # Critic loss: Regress toward target return
            crit_loss = F.mse_loss(val_imag, target_vals.detach())
            
            # Optimizer steps
            optimizer_actor.zero_grad()
            act_loss.backward()
            
            optimizer_critic.zero_grad()
            crit_loss.backward()
            
            optimizer_actor.step()
            optimizer_critic.step()
            
            total_loss_actor += act_loss.item()
            
            # Reporting
            if time.time() - last_report_time > 1.0 or i == 0:
                last_report_time = time.time()
                if progress_callback: progress_callback(epoch + (i + 1) / num_batches, wm_loss.item())

        avg_loss = total_loss_wm / max(1, len(dataloader))
        logger.info(f"Epoch {epoch+1}/{num_epochs} COMPLETE, Avg WM Loss: {avg_loss:.4f}")
        if progress_callback: progress_callback(epoch + 1, avg_loss)
        
    # Save Model
    if model_filename:
        if not model_filename.endswith('.pth'): model_filename += '.pth'
        policy_basename = model_filename.replace('.pth', '')
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        policy_basename = f"{tag}-dreamer_{timestamp}"
        
    policy_path = os.path.join(data_root, f"{policy_basename}.pth")
    
    save_dict = {
        'model_state_dict': dreamer.state_dict(),
        'latent_dim': latent_dim_det,
        'model_size': model_size_det
    }
    torch.save(save_dict, policy_path)
    logger.info(f"Dreamer Model saved: {policy_path}")
    
    return policy_path
