import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime

# Add the parent directories to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.models.vint import ViNTModel
from backend.training.datasets.vint_dataset import ViNTDataset

def train(epochs, batch_size, lr, data_root, context_size, freeze_backbone):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load transitions
    transitions_path = os.path.join(data_root, 'all_transitions.json')
    if not os.path.exists(transitions_path):
        print(f"Error: Could not find {transitions_path}")
        return

    with open(transitions_path, 'r') as f:
        transitions = json.load(f)

    # Initialize Dataset
    dataset = ViNTDataset(
        transitions, 
        data_root=data_root, 
        context_size=context_size,
        max_lookahead=20,
        device=str(device)
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize Model
    # 4 Actions: Forward, Reverse, Left, Right
    model = ViNTModel(context_size=context_size, num_actions=4, freeze_backbone=freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Filter parameters to only train those that require gradients
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('backend/models/checkpoints', exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for obs_hist, goal_img, action in pbar:
            obs_hist = obs_hist.to(device)
            goal_img = goal_img.to(device)
            action = action.to(device)

            optimizer.zero_grad()

            logits = model(obs_hist, goal_img)
            loss = criterion(logits, action)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == action).sum().item()
            total_samples += action.size(0)

            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total_samples})

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f} - Avg Acc: {avg_acc:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = f"backend/models/checkpoints/vint_{timestamp}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--context-size', type=int, default=3, help="Number of observation frames to use as context history")
    parser.add_argument('--unfreeze', action='store_true', help="Unfreeze the EfficientNet backbone")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_root=args.data_root,
        context_size=args.context_size,
        freeze_backbone=not args.unfreeze
    )
