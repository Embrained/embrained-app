
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy
import cv2
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Import new Spatial Models
from modules.spatial_model import SpatialEncoder, CQLNetwork, FullQNet

# Configure Logging
logger = logging.getLogger("TrainSpatialCQL")
logging.basicConfig(level=logging.INFO)

# --- Hyperparameters ---
BATCH_SIZE = 32 # Detailed images take more VRAM
GAMMA = 0.99
LEARNING_RATE = 1e-4
CQL_ALPHA = 1.0
COSINE_ALPHA = 1.0 # Reward scaling
NUM_EPOCHS = 5
HIDDEN_DIM = 256
INPUT_DIM = 128 # 64 kpt current + 64 kpt goal
ACTION_DIM = 5 # 0:Stop, 1:Fwd, 2:Left, 3:Right, 4:Back

IMG_H = 120
IMG_W = 160

# Transform for MobileNet Backbone (Stage A)
# Note: Input size is 120x160, not 224x224
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def discretize_action(left, right):
    tol = 10
    if abs(left) < tol and abs(right) < tol: return 0
    if left > tol and right > tol: return 1 # Fwd
    if left < -tol and right > tol: return 2 # Left
    if left < -tol and right < -tol: return 3 # Back (Aligned with Engine default for Manual but check mapping)
    if left > tol and right < -tol: return 4 # Right
    return 0

class SpatialCQLDataset(Dataset):
    def __init__(self, episodes, data_root, device, progress_callback=None):
        self.samples = []
        self.data_root = data_root
        self.device = device
        
        logger.info("SpatialCQLDataset: Verifying data integrity (checking file paths)...")
        
        missing_count = 0
        total_eps = len(episodes)
        
        for idx, ep in enumerate(episodes):
            if progress_callback and idx % 10 == 0:
                progress_callback(0, float(idx)/total_eps) # Epoch 0 for setup phase
                
            full_seq = [ep['start_frame']] + ep['actions']
            goal_node = ep['goal_frame']
            
            # Goal Image Check
            # Prepend data_root to ensure absolute/correct relative path
            goal_path = os.path.join(data_root, goal_node['image_path'])
            if not os.path.exists(goal_path): 
                missing_count += 1
                continue

            for i in range(len(full_seq) - 1):
                curr_node = full_seq[i]
                next_node = full_seq[i+1]
                
                curr_path = os.path.join(data_root, curr_node['image_path'])
                next_path = os.path.join(data_root, next_node['image_path'])
                
                if not os.path.exists(curr_path) or not os.path.exists(next_path):
                    missing_count += 1
                    continue
                    
                l_cmd = next_node['left_cmd']
                r_cmd = next_node['right_cmd']
                action = discretize_action(l_cmd, r_cmd)
                
                is_last_step = (i == len(full_seq) - 2)
                reward = 1.0 if is_last_step else 0.0 # Simple sparse reward
                done = is_last_step
                
                self.samples.append({
                    'curr_path': curr_path,
                    'next_path': next_path,
                    'goal_path': goal_path,
                    'action': action,
                    'reward': reward,
                    'done': done
                })
                
        if missing_count > 0:
            logger.warning(f"Skipped {missing_count} transitions due to missing files (or bad paths).")
            
        # --- SUBSAMPLING FOR SPEED (Smoke Test) ---
        MAX_SAMPLES = 2000
        if len(self.samples) > MAX_SAMPLES:
            logger.warning(f"SUBSAMPLING: Reducing dataset from {len(self.samples)} to {MAX_SAMPLES} for speed.")
            import random
            random.shuffle(self.samples)
            self.samples = self.samples[:MAX_SAMPLES]
            
    def _load_img(self, path):
        # Open with OpenCV, convert to RGB
        img = cv2.imread(path)
        if img is None:
            # Fallback black image
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return transform(img) # Tensor (3, 120, 160)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        img_curr = self._load_img(s['curr_path'])
        img_next = self._load_img(s['next_path'])
        img_goal = self._load_img(s['goal_path'])
        
        return (
            img_curr,
            img_next,
            img_goal,
            torch.LongTensor([s['action']]),
            torch.FloatTensor([s['reward']]),
            torch.FloatTensor([1.0 if s['done'] else 0.0])
        )

def train(data_root, num_epochs=NUM_EPOCHS, stop_event=None, progress_callback=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device} (CUDA Available: {torch.cuda.is_available()})")
    
    # Load Episodes
    ep_path = os.path.join(data_root, "episodes.json")
    if not os.path.exists(ep_path):
        raise FileNotFoundError(f"episodes.json not found in {data_root}")
        
    with open(ep_path, 'r') as f:
        episodes = json.load(f)
        
    # Prepare Dataset
    if progress_callback: progress_callback(0, 0.1) # Signal start
    logger.info("Preparing Dataset (this may take a while)...")
    dataset = SpatialCQLDataset(episodes, data_root, device, progress_callback)
    if progress_callback: progress_callback(0, 1.0) # Signal done
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 0 for simplicity/safety on Windows
    
    # Init Models
    encoder = SpatialEncoder(output_keypoints=32, frozen_backbone=True).to(device)
    policy = CQLNetwork(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM).to(device)
    
    q_net = FullQNet(encoder, policy).to(device)
    target_q_net = copy.deepcopy(q_net) # Copy full structure
    
    # Optimizer: Only train adapter and policy
    # encoder.backbone matches 'backbone' name in SpatialEncoder
    params_to_train = [
        {'params': encoder.adapter_conv.parameters()},
        {'params': policy.parameters()}
    ]
    optimizer = optim.Adam(params_to_train, lr=LEARNING_RATE)
    
    q_net.train() # This sets training mode. 
    # But backbone is frozen via requires_grad=False in Init.
    # And we filtered optimizer params.
    # Important: Backbone BatchNorm layers?
    # Usually we want eval() for frozen backbone to keep statistics fixed if pre-trained.
    q_net.encoder.backbone.eval()
    
    logger.info(f"Starting Training Loop for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Stop Check
        if stop_event and stop_event.is_set():
            logger.info("Training interrupted by user.")
            break

        total_loss = 0
        num_batches = len(dataloader)
        
        for i, (img_curr, img_next, img_goal, action, reward, done) in enumerate(dataloader):
            # Intra-batch stop check
            if stop_event and stop_event.is_set():
                break
                
            img_curr = img_curr.to(device)
            img_next = img_next.to(device)
            img_goal = img_goal.to(device)
            action = action.to(device)
            reward = reward.to(device)
            done = done.to(device)
            
            # Current Q: (Batch, ActionDim)
            q_values = q_net(img_curr, img_goal) 
            q_action = q_values.gather(1, action)
            
            # Target Q
            with torch.no_grad():
                # Double DQN / Standard DQN
                next_q_values = target_q_net(img_next, img_goal)
                max_next_q = next_q_values.max(1, keepdim=True)[0]
                target_q = reward + GAMMA * (1 - done) * max_next_q
                
            # TD Loss
            td_loss = F.mse_loss(q_action, target_q)
            
            # CQL Loss
            logsumexp_q = torch.logsumexp(q_values, dim=1, keepdim=True)
            cql_loss = (logsumexp_q - q_action).mean()
            
            loss = td_loss + CQL_ALPHA * cql_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {i+1}/{num_batches} | Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / max(1, len(dataloader))
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} COMPLETE, Avg Loss: {avg_loss:.4f}")
        
        # Dial home
        if progress_callback:
            progress_callback(epoch + 1, avg_loss)
        
        # Target Update
        target_q_net.load_state_dict(q_net.state_dict())
        
    # Save Models
    # We need to save Encoder and Policy separately for inference usage pattern
    # (VisionSystem loads Encoder, Planner loads Policy)
    
    # Determine save directory (data_root usually)
    encoder_path = os.path.join(data_root, "spatial_encoder.pth")
    policy_path = os.path.join(data_root, "cql_policy.pth")
    
    torch.save(q_net.encoder.state_dict(), encoder_path)
    torch.save(q_net.policy.state_dict(), policy_path)
    
    logger.info(f"Models saved: {encoder_path}, {policy_path}")
    
    # Generate Goals (Optional but helpful for Planner)
    generate_goals(q_net.encoder, episodes, data_root, device)
    
    return encoder_path, policy_path

def generate_goals(encoder, episodes, data_root, device):
    """
    Extracts unique goal images, encodes them to keypoints, and saves goals.npy
    """
    logger.info("Generating Goal Keypoints...")
    goal_paths = set()
    for ep in episodes:
        goal_path = os.path.join(data_root, ep['goal_frame']['image_path'])
        goal_paths.add(goal_path)
    
    # Cap goals to avoid hanging on massive datasets
    MAX_GOALS = 200
    goal_paths_list = list(goal_paths)
    if len(goal_paths_list) > MAX_GOALS:
        logger.info(f"Subsampling goals from {len(goal_paths_list)} to {MAX_GOALS} for speed.")
        import random
        random.shuffle(goal_paths_list)
        goal_paths_list = goal_paths_list[:MAX_GOALS]
        
    goals = []
    encoder.eval()
    with torch.no_grad():
        for gp in goal_paths_list:
            if os.path.exists(gp):
                img = cv2.imread(gp)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = transform(img).unsqueeze(0).to(device)
                    kpts = encoder(tensor) # (1, 64)
                    goals.append(kpts.cpu().numpy().flatten())
                    
    if goals:
        save_path = os.path.join(data_root, "goals.npy")
        np.save(save_path, np.array(goals))
        logger.info(f"Saved {len(goals)} goals to {save_path}")

if __name__ == "__main__":
    # Assuming script run from backend/ or project root
    # Try to locate data dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "../data")
    if not os.path.exists(data_dir):
        # Fallback for if running from embrained-app root
        data_dir = os.path.join(base_dir, "data")
        
    train(data_dir)
