import os
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import logging
import random

logger = logging.getLogger("ViNTDataset")

class ViNTDataset(Dataset):
    def __init__(self, transitions, data_root, context_size=3, max_lookahead=20, device='cpu', image_size=224):
        self.samples = []
        self.data_root = data_root
        self.device = device
        self.video_caps = {}
        self.image_size = image_size
        self.context_size = context_size
        self.max_lookahead = max_lookahead
        
        # EfficientNet requires standard ImageNet normalization
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"ViNTDataset: Initializing with {len(transitions)} raw transitions...")

        # Group by session
        sessions = {}
        for t in transitions:
            s = t.get('session', 'default')
            if s not in sessions:
                sessions[s] = []
            sessions[s].append(t)
            
        for s, traj in sessions.items():
            traj_sorted = sorted(traj, key=lambda x: x.get('timestamp', 0))
            n = len(traj_sorted)
            if n < 2:
                continue
                
            for t in range(n - 1):
                curr_node = traj_sorted[t]
                
                # Action mapping (0: Forward, 1: Reverse, 2: Left, 3: Right)
                macro_action = curr_node.get('macro_action', 0)
                if macro_action == 1:
                    action_idx = 0
                elif macro_action == 2:
                    action_idx = 1
                elif macro_action == 3:
                    action_idx = 2
                elif macro_action == 4:
                    action_idx = 3
                else:
                    # Skip stops (0) or unknown actions
                    continue
                
                # Create history context buffer
                context_nodes = []
                start_idx = t - self.context_size + 1
                for idx in range(start_idx, t + 1):
                    safe_idx = max(0, idx)  # Pad with first frame if t is too small
                    context_nodes.append(traj_sorted[safe_idx])
                
                # Sample a goal node from the future
                max_g = min(n - 1, t + self.max_lookahead)
                if max_g <= t: 
                    continue # no future frames
                    
                g = random.randint(t + 1, max_g)
                goal_node = traj_sorted[g]
                
                self.samples.append({
                    'context_nodes': context_nodes,
                    'goal_node': goal_node,
                    'action': action_idx
                })
                
        random.shuffle(self.samples)
        logger.info(f"Generated {len(self.samples)} context-goal pairs for ViNT.")

    def __del__(self):
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
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return cv2.resize(frame_rgb, (self.image_size, self.image_size))
                return None

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
            if 'image_path' in node:
                p = node['image_path']
                if 'webcam_' not in p:
                    if not os.path.isabs(p):
                        p = os.path.join(self.data_root, p)
                    if os.path.exists(p):
                        img = cv2.imread(p)
                        if img is not None:
                            return self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if node.get('format') == 'lerobot' and 'video_path' in node:
                vid_path = node['video_path']
                if 'webcam' not in vid_path:
                    if not os.path.isabs(vid_path):
                        vid_path = os.path.join(self.data_root, vid_path)
                    ds_path = os.path.dirname(vid_path)
                    if os.path.basename(ds_path) == "videos":
                         ds_path = os.path.dirname(ds_path)
                    
                    frame_idx = node['frame_idx']
                    img_path = os.path.join(ds_path, "images", f"frame_{frame_idx:06d}.jpg")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            return self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
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
        
        # Load context images
        context_imgs = [self._load_img(node) for node in sample['context_nodes']]
        obs_hist = torch.stack(context_imgs) # (H, 3, 224, 224)
        
        # Load goal image
        goal_img = self._load_img(sample['goal_node']) # (3, 224, 224)
        
        action = torch.tensor(sample['action'], dtype=torch.long)

        return obs_hist, goal_img, action
