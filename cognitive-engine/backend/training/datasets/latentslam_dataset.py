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
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import logging
import random
import math

logger = logging.getLogger("LatentSLAMDataset")

class LatentSLAMDataset(Dataset):
    def __init__(self, transitions, data_root, device='cpu', progress_callback=None, image_size=64):
        self.samples = []
        self.data_root = data_root
        self.device = device
        self.video_caps = {}
        self.image_size = image_size
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])
        
        logger.debug(f"LatentSLAMDataset: Initializing with {len(transitions)} raw transitions...")

        # We need pairs (s_t, a_t, s_{t+1}).
        # Group by session and sort by time, then pair adjacent steps.
        sessions = {}
        for t in transitions:
            s = t.get('session', 'default')
            if s not in sessions:
                sessions[s] = []
            sessions[s].append(t)
            
        for s, traj in sessions.items():
            traj_sorted = sorted(traj, key=lambda x: x.get('timestamp', 0))
            for i in range(len(traj_sorted) - 1):
                curr_node = traj_sorted[i]
                next_node = traj_sorted[i+1]
                
                # We use discrete valid actions (Forward=1, Left=3, Right=4)
                # Map them to contiguous indices: Fwd=0, Left=1, Right=2
                macro_action = curr_node.get('macro_action', 0)
                if macro_action == 1:
                    action_idx = 0
                elif macro_action == 3:
                    action_idx = 1
                elif macro_action == 4:
                    action_idx = 2
                else:
                    # Skip stops (0), reverse (2), or unknown actions
                    continue
                    
                self.samples.append({
                    'curr_node': curr_node,
                    'action': action_idx,
                    'next_node': next_node
                })
                
        # Shuffle explicitly here or rely on DataLoader
        random.shuffle(self.samples)
        logger.debug(f"Generated {len(self.samples)} transition pairs for LatentSLAM.")

    def __del__(self):
        """Release all video captures on cleanup."""
        if hasattr(self, 'video_caps'):
            for path, obj in self.video_caps.items():
                if hasattr(obj, 'isOpened') and obj.isOpened():
                    obj.release()
            self.video_caps.clear()

    def _get_frame_from_video(self, video_path, frame_idx):
        if not os.path.exists(video_path):
            return None

        if video_path not in self.video_caps:
            MAX_RAM_VIDEOS = 200
            if len(self.video_caps) >= MAX_RAM_VIDEOS:
                # Direct seek
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return cv2.resize(frame_rgb, (self.image_size, self.image_size))
                return None

            # Preload video into RAM
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
                
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
                frames.append(frame_resized)
                
            cap.release()
            self.video_caps[video_path] = frames
            
        frames = self.video_caps[video_path]
        if frame_idx < len(frames):
            return frames[frame_idx]
            
        return None

    def _load_img(self, node):
        if not node:
            return torch.zeros((3, self.image_size, self.image_size))
            
        try:
            # 1. Check for explicit image path (markov pattern)
            if 'image_path' in node:
                p = node['image_path']
                # Ignore newly added external camera images
                if 'webcam_' not in p:
                    if not os.path.isabs(p):
                        p = os.path.join(self.data_root, p)
                    
                    if os.path.exists(p):
                        img = cv2.imread(p)
                        if img is not None:
                            return self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # 2. Check for LeRobot Video + Index
            if node.get('format') == 'lerobot' and 'video_path' in node:
                vid_path = node['video_path']
                # Ignore videos labeled as webcam streams
                if 'webcam' not in vid_path:
                    if not os.path.isabs(vid_path):
                        vid_path = os.path.join(self.data_root, vid_path)
                    ds_path = os.path.dirname(vid_path)
                    # Check for videos wrapper
                    if os.path.basename(ds_path) == "videos":
                         ds_path = os.path.dirname(ds_path)
                    
                    frame_idx = node['frame_idx']
                    img_path = os.path.join(ds_path, "images", f"frame_{frame_idx:06d}.jpg")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            return self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
                    # Fallback to video
                    frame = self._get_frame_from_video(vid_path, frame_idx)
                    if frame is not None:
                        return self.transform(frame)
                    
        except Exception as e:
            logger.error(f"Error loading image for node: {e}")
            
        return torch.zeros((3, self.image_size, self.image_size))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_curr = self._load_img(sample['curr_node'])
        img_next = self._load_img(sample['next_node'])
        action = torch.tensor(sample['action'], dtype=torch.long)

        return img_curr, action, img_next
