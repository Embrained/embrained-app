import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CNN Regressor mapping Image -> (x, y, cos, sin)
class PoseRegressor(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, n_layers=4):
        super(PoseRegressor, self).__init__()
        
        modules = []
        current_channels = base_channels
        
        modules.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, stride=1, padding=1))
        modules.append(nn.ReLU())
        
        for _ in range(n_layers):
            out_channels = min(current_channels * 2, 512)
            modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            modules.append(nn.ReLU())
            current_channels = out_channels
            
        modules.append(nn.Flatten())
        
        # 64x64 input -> downsampled 4 times -> 4x4 spatial dim
        # 4 * 4 * current_channels (which is 512) = 8192
        self.encoder = nn.Sequential(*modules)
        
        self.regressor = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # cx, cy, cos_yaw, sin_yaw
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.regressor(feat)

class PoseDataset(Dataset):
    def __init__(self, data_root, split='train'):
        self.data_root = data_root
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()
        ])
        
        # Load Telemetry
        self.telemetry = {}
        master_path = os.path.join(data_root, "master_telemetry.csv")
        if not os.path.exists(master_path):
            raise Exception("No telemetry found")
            
        df = pd.read_csv(master_path)
        for _, row in df.iterrows():
            try:
                yaw_rad = math.radians(row['yaw_deg'])
                self.telemetry[str(row['ts'])] = [
                    float(row['cx']) / 640.0,
                    float(row['cy']) / 480.0,
                    math.cos(yaw_rad),
                    math.sin(yaw_rad)
                ]
            except KeyError:
                continue
                
        # Load Images mapping
        trans_path = os.path.join(data_root, "all_transitions.json")
        with open(trans_path, 'r') as f:
            all_data = json.load(f)
            
        self.samples = []
        for item in all_data:
            img_path = item.get('image_path', '')
            if not img_path: continue
            
            ts = os.path.basename(img_path).replace('frame_', '').replace('.jpg', '')
            if ts in self.telemetry:
                self.samples.append({
                    'image_path': img_path,
                    'pose': self.telemetry[ts]
                })
                
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * 0.8)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
            
        print(f"Loaded {len(self.samples)} samples for {split} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = str(item['image_path']).replace('\\', '/')
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.data_root, img_path)
        
        # fallback fix if image missing
        if not os.path.exists(img_path):
             for search_root in [self.data_root, os.path.join(self.data_root, '..')]:
                 attempt = os.path.join(search_root, item['image_path'])
                 if os.path.exists(attempt):
                     img_path = attempt
                     break
                     
        img = Image.open(img_path).convert('RGB')
        tensor_img = self.transform(img)
        pose = torch.tensor(item['pose'], dtype=torch.float32)
        return tensor_img, pose

def train():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Setting up Regression Datasets...")
    train_dataset = PoseDataset(data_root, split='train')
    val_dataset = PoseDataset(data_root, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    model = PoseRegressor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 15
    train_losses = []
    val_losses = []
    
    print("Starting Absolute Pose Regression Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            
            # Weighted loss - give more weight to XY than rotation
            loss_xy = criterion(preds[:, :2], y[:, :2]) * 2.0
            loss_rot = criterion(preds[:, 2:], y[:, 2:])
            loss = loss_xy + loss_rot
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Valid
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                v_loss_xy = criterion(preds[:, :2], y[:, :2]) * 2.0
                v_loss_rot = criterion(preds[:, 2:], y[:, 2:])
                val_loss += (v_loss_xy + v_loss_rot).item()
                
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
    # Plot results
    plt.figure()
    plt.plot(range(epochs), train_losses, label='Train MSE')
    plt.plot(range(epochs), val_losses, label='Val MSE')
    plt.title('Absolute Pose Regression Overfitting vs Generalization')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (XY heavily weighted)')
    plt.legend()
    plt.savefig(os.path.join(data_root, 'pose_regression_curve.png'))
    plt.close()
    
    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(data_root, 'absolute_pose_regressor.pth'))
    print("Training complete. Model saved. Check the curve for catastrophic domain overfitting.")
    
if __name__ == "__main__":
    train()
