import os
import sys
import json
import math
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import ACTION_PWM_MAP, DATA_DIR

T_HORIZON = 10 # Number of steps preceding the goal image to count as "successful approach trajectory"

TARGET_IMAGES = []
goals_dir = os.path.join(DATA_DIR, 'goals')
if os.path.exists(goals_dir):
    for f in os.listdir(goals_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            TARGET_IMAGES.append(f"goals/{f}")

if not TARGET_IMAGES:
    print("Warning: No images found in data/goals. Falling back to hardcoded goals.")
    TARGET_IMAGES = [
        "markov_2026-03-22_13-34-29/images/frame_1774201370056.jpg",
        "markov_2026-03-23_17-13-37/images/frame_1774300483517.jpg",
        "markov_2026-03-23_17-13-37/images/frame_1774300685051.jpg",
        "markov_2026-03-28_15-36-36/images/frame_1774726629342.jpg",
        "markov_2026-03-28_15-36-36/images/frame_1774727221978.jpg",
        "markov_2026-04-09_17-57-21/images/frame_1775771989581.jpg",
        "markov_2026-04-09_19-01-19/images/frame_1775775762862.jpg",
        "markov_2026-04-10_10-03-48/images/frame_1775829945267.jpg",
        "markov_2026-04-10_10-03-48/images/frame_1775830001252.jpg",
        "markov_2026-04-10_10-03-48/images/frame_1775830792618.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776300989512.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776301260549.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776301558193.jpg",
        "markov_2026-04-15_20-45-54/images/frame_1776301789015.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776298380051.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776298778840.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776298791319.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776299291559.jpg",
        "markov_2026-04-15_20-08-29/images/frame_1776299387708.jpg",
        "markov_2026-04-15_19-52-06/images/frame_1776297443526.jpg",
        "markov_2026-04-15_19-52-06/images/frame_1776297814967.jpg",
        "markov_2026-04-15_19-52-06/images/frame_1776298091052.jpg",
        "markov_2026-04-15_17-17-37/images/frame_1776288722567.jpg",
        "markov_2026-04-15_17-17-37/images/frame_1776289155128.jpg",
        "markov_2026-04-15_17-17-37/images/frame_1776289567346.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287001218.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287067469.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287320412.jpg",
        "markov_2026-04-15_16-51-25/images/frame_1776287321847.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285752462.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285787367.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285794981.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776285807256.jpg",
        "markov_2026-04-15_16-31-17/images/frame_1776286083024.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283828658.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283855887.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283954649.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776283949674.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776284138608.jpg",
        "markov_2026-04-15_16-09-06/images/frame_1776284184794.jpg"
    ]

def normalize_path(p):
    if not p: return ""
    return str(p).replace('/', '\\').split('data\\')[-1]

class EndToEndBCDataset(Dataset):
    def __init__(self, data_root, transitions):
        self.data_root = data_root
        self.transitions = transitions
        # ImageNet standardization
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        item = self.transitions[idx]
        
        img_path = os.path.join(self.data_root, item.get('image_path', ''))
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
        except Exception as e:
            # Fallback
            img_tensor = torch.zeros((3, 224, 224))
            
        action = item.get('action_id', 5) # Default to STOP
        return img_tensor, torch.tensor(action, dtype=torch.long)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting End-To-End Behavioral Cloning on {device}")
    
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    trans_path = os.path.join(DATA_ROOT, "all_transitions.json")
    if not os.path.exists(trans_path):
         print(f"Error: {trans_path} not found.")
         return
         
    with open(trans_path, 'r') as f:
        all_data = json.load(f)
        
    sessions = {}
    for item in all_data:
        s = item['session']
        if s not in sessions: sessions[s] = []
        sessions[s].append(item)
        
    target_filenames = set(os.path.basename(p) for p in TARGET_IMAGES)
    
    expert_transitions = []
    print("Isolating trajectories ending in goal images...")
    
    for s_name, traj in sessions.items():
        traj = sorted(traj, key=lambda x: x['timestamp'])
        
        for i, node in enumerate(traj):
            p = normalize_path(node.get('image_path', ''))
            filename = os.path.basename(p)
            
            # Simple matching: Does this node path's filename match any target?
            if filename in target_filenames:
                # Tag the goal frame as STOP (Action 5)
                goal_node = node.copy()
                goal_node['action_id'] = 5
                expert_transitions.append(goal_node)
                
                # Extract preceding T_HORIZON frames
                start_idx = max(0, i - T_HORIZON)
                for j in range(start_idx, i):
                    prev_node = traj[j].copy()
                    
                    # Resolve action taken at prev_node
                    best_action = 1 # Fwd default
                    if 'macro_action' in prev_node:
                        best_action = int(prev_node['macro_action'])
                    else:
                        raw_l = float(prev_node.get('left_cmd', 0.0))
                        raw_r = float(prev_node.get('right_cmd', 0.0))
                        best_dist = float('inf')
                        for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
                            dist = math.hypot(raw_l - map_l, raw_r - map_r)
                            if dist < best_dist:
                                best_dist = dist
                                best_action = act_id
                    
                    prev_node['action_id'] = best_action
                    expert_transitions.append(prev_node)
    
    # Remove duplicates based on image path
    unique_transitions = {}
    for t in expert_transitions:
        p = t.get('image_path', '')
        if p and p not in unique_transitions:
            unique_transitions[p] = t
            
    expert_transitions = list(unique_transitions.values())
    print(f"Compiled {len(expert_transitions)} expert trajectory frames.")
    
    if len(expert_transitions) == 0:
        print("No expert transitions found! Check target image lists.")
        return
        
    dataset = EndToEndBCDataset(DATA_ROOT, expert_transitions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize ResNet18
    print("Initializing ResNet18 ImageNet weights...")
    # Use weights parameter instead of pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze lower layers to prevent catastrophic forgetting and speed up training
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last block
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace the FC layer to output 6 action logits
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    model = model.to(device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    print("Beginning Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        
    out_path = os.path.join(DATA_ROOT, "e2e_tv_model.pth")
    torch.save(model.state_dict(), out_path)
    print(f"End-to-End model saved to: {out_path}")

if __name__ == "__main__":
    train()
