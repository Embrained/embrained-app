# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Training script for the Sofa Proximity Classifier.
# Trains a binary CNN to distinguish "sofa close-up" from "not sofa close-up"
# using user-curated positive examples in data/sofa/.

import os
import sys
import glob
import random
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cognitive-engine')))
from config import DATA_DIR
from modules.goal_classifier import GoalClassifier


class GoalClassifierDataset(Dataset):
    """Dataset for binary goal classification.
    
    Positive examples: All images in the goal directory (e.g., data/sofa/)
    Negative examples: Random sample from exploration data (data/markov_*/images/)
    """
    
    def __init__(self, positive_paths, negative_paths, img_dim=64, augment=True):
        self.img_dim = img_dim
        self.augment = augment
        
        # Build samples list: (path, label)
        self.samples = []
        for p in positive_paths:
            self.samples.append((p, 1))
        for p in negative_paths:
            self.samples.append((p, 0))
        
        random.shuffle(self.samples)
        
        # Base transform (always applied)
        self.base_transform = T.Compose([
            T.Resize((img_dim, img_dim)),
            T.ToTensor(),
        ])
        
        # Augmentation transform (for training positives to increase variety)
        self.aug_transform = T.Compose([
            T.Resize((img_dim, img_dim)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Return a black image on failure
            img = Image.new('RGB', (self.img_dim, self.img_dim), (0, 0, 0))
        
        # Apply augmentation to positives during training
        if self.augment and label == 1:
            tensor = self.aug_transform(img)
        else:
            tensor = self.base_transform(img)
        
        return tensor, torch.tensor(label, dtype=torch.float32)


def collect_image_paths(data_root, goal_subdir='sofa'):
    """Collect positive and negative image paths."""
    
    # Positive: all images in the goal subdirectory
    goal_dir = os.path.join(data_root, goal_subdir)
    if not os.path.isdir(goal_dir):
        raise FileNotFoundError(f"Goal directory not found: {goal_dir}")
    
    positive_paths = sorted(glob.glob(os.path.join(goal_dir, '*.jpg')))
    positive_basenames = set(os.path.basename(p) for p in positive_paths)
    
    print(f"Found {len(positive_paths)} positive examples in {goal_dir}")
    
    # Negative: all images from exploration sessions
    negative_paths = []
    session_dirs = sorted(glob.glob(os.path.join(data_root, 'markov_*')))
    
    for session_dir in session_dirs:
        img_dir = os.path.join(session_dir, 'images')
        if not os.path.isdir(img_dir):
            continue
        
        for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
            basename = os.path.basename(img_path)
            # Exclude any frame that's also in the positive set
            if basename not in positive_basenames:
                negative_paths.append(img_path)
    
    print(f"Found {len(negative_paths)} negative candidate frames from {len(session_dirs)} sessions")
    
    return positive_paths, negative_paths


def train_classifier(data_root, goal_subdir='sofa', model_size='large', img_dim=64,
                     num_epochs=30, batch_size=64, lr=1e-3, val_split=0.2,
                     max_neg_ratio=5.0):
    """Train a binary GoalClassifier.
    
    Args:
        data_root: Path to data directory
        goal_subdir: Subdirectory name for positive examples (e.g., 'sofa', 'tv')
        model_size: TinyVAE backbone size
        img_dim: Input image dimension
        num_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split fraction
        max_neg_ratio: Max ratio of negatives to positives (for manageable training)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f" GOAL CLASSIFIER TRAINING ({goal_subdir.upper()})")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # 1. Collect data
    positive_paths, negative_paths = collect_image_paths(data_root, goal_subdir)
    
    if len(positive_paths) < 10:
        print(f"ERROR: Only {len(positive_paths)} positive examples found. Need at least 10.")
        return None
    
    # Limit negatives to avoid massive class imbalance in the dataset
    max_negatives = int(len(positive_paths) * max_neg_ratio)
    if len(negative_paths) > max_negatives:
        random.shuffle(negative_paths)
        negative_paths = negative_paths[:max_negatives]
        print(f"Subsampled negatives to {len(negative_paths)} (ratio {max_neg_ratio}:1)")
    
    # 2. Train/val split (stratified)
    random.shuffle(positive_paths)
    random.shuffle(negative_paths)
    
    n_pos_val = max(1, int(len(positive_paths) * val_split))
    n_neg_val = max(1, int(len(negative_paths) * val_split))
    
    pos_train, pos_val = positive_paths[n_pos_val:], positive_paths[:n_pos_val]
    neg_train, neg_val = negative_paths[n_neg_val:], negative_paths[:n_neg_val]
    
    print(f"\nTrain: {len(pos_train)} pos + {len(neg_train)} neg = {len(pos_train)+len(neg_train)}")
    print(f"Val:   {len(pos_val)} pos + {len(neg_val)} neg = {len(pos_val)+len(neg_val)}")
    
    train_dataset = GoalClassifierDataset(pos_train, neg_train, img_dim=img_dim, augment=True)
    val_dataset = GoalClassifierDataset(pos_val, neg_val, img_dim=img_dim, augment=False)
    
    # 3. Weighted sampler for balanced batches
    n_pos = len(pos_train)
    n_neg = len(neg_train)
    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg
    sample_weights = []
    for _, label in train_dataset.samples:
        sample_weights.append(weight_pos if label == 1 else weight_neg)
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 4. Model
    model = GoalClassifier(
        latent_dim=32,
        model_size=model_size,
        input_spatial_dim=img_dim,
        in_channels=3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: GoalClassifier ({model_size}), {total_params:,} params")
    
    # 5. Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    
    best_f1 = 0.0
    best_state = None
    best_metrics = {}
    train_losses = []
    val_f1s = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        scheduler.step()
        
        # --- Validate ---
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > 0.5).long()
                
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.long().tolist())
                all_probs.extend(probs.tolist())
        
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        val_f1s.append(val_f1)
        
        # Log every 5 epochs or on improvement
        if (epoch + 1) % 5 == 0 or val_f1 > best_f1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Track best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                'epoch': epoch + 1,
                'f1': val_f1,
                'loss': avg_loss,
            }
    
    print("-" * 60)
    print(f"Best Val F1: {best_f1:.4f} (epoch {best_metrics.get('epoch', '?')})")
    
    # 6. Final evaluation with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.long().tolist())
            all_probs.extend(probs.tolist())
    
    print("\n" + "=" * 40)
    print(" FINAL VALIDATION RESULTS")
    print("=" * 40)
    print(classification_report(all_labels, all_preds, target_names=['Not-Goal', goal_subdir.title()], zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    
    # 7. Save model
    save_path = os.path.join(data_root, f'{goal_subdir}_classifier.pth')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_size': model_size,
        'latent_dim': 32,
        'input_spatial_dim': img_dim,
        'in_channels': 3,
        'threshold': 0.5,
        'goal_subdir': goal_subdir,
        'metrics': best_metrics,
        'num_positive': len(positive_paths),
        'num_negative': len(negative_paths),
    }
    torch.save(checkpoint, save_path)
    print(f"\n[OK] Saved classifier to {save_path}")
    
    # 8. Generate prediction visualization grid
    _generate_prediction_grid(model, val_dataset, device, goal_subdir, data_root)
    
    # 9. Plot training curves
    _plot_training_curves(train_losses, val_f1s, goal_subdir)
    
    return save_path


def _generate_prediction_grid(model, val_dataset, device, goal_subdir, data_root):
    """Generate a visual grid of top predictions for sanity checking."""
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            img_tensor, label = val_dataset[i]
            logit = model(img_tensor.unsqueeze(0).to(device))
            prob = torch.sigmoid(logit).item()
            path = val_dataset.samples[i][0]
            results.append((prob, label.item(), path, img_tensor))
    
    # Sort by confidence
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Create grid: top 5 most confident positive, top 5 most confident negative
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'{goal_subdir.title()} Classifier — Top Predictions', fontsize=14, fontweight='bold')
    
    # Top row: highest confidence (should be positive)
    for i in range(min(5, len(results))):
        prob, label, path, tensor = results[i]
        img = tensor.permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        color = 'green' if label == 1 else 'red'
        axes[0, i].set_title(f'P={prob:.2f}\n{"✓" if label==1 else "✗"} GT={int(label)}', color=color, fontsize=10)
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Highest\nConfidence', fontsize=11, fontweight='bold', rotation=0, labelpad=60)
    
    # Bottom row: lowest confidence (should be negative)
    for i in range(min(5, len(results))):
        prob, label, path, tensor = results[-(i+1)]
        img = tensor.permute(1, 2, 0).numpy()
        axes[1, i].imshow(img)
        color = 'green' if label == 0 else 'red'
        axes[1, i].set_title(f'P={prob:.2f}\n{"✓" if label==0 else "✗"} GT={int(label)}', color=color, fontsize=10)
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('Lowest\nConfidence', fontsize=11, fontweight='bold', rotation=0, labelpad=60)
    
    plt.tight_layout()
    
    images_dir = os.path.join(os.path.dirname(data_root), 'images')
    os.makedirs(images_dir, exist_ok=True)
    save_path = os.path.join(images_dir, f'{goal_subdir}_classifier_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved prediction grid to {save_path}")


def _plot_training_curves(train_losses, val_f1s, goal_subdir):
    """Plot training loss and validation F1 curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{goal_subdir.title()} Classifier — Training Curves', fontsize=13, fontweight='bold')
    
    ax1.plot(train_losses, color='#FF6B6B', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_f1s, color='#4ECDC4', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='Target (0.90)')
    ax2.legend()
    
    plt.tight_layout()
    
    images_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images'))
    os.makedirs(images_dir, exist_ok=True)
    save_path = os.path.join(images_dir, f'{goal_subdir}_classifier_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved training curves to {save_path}")


def main():
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    # Check for goal subdirectory
    goal_subdir = 'sofa'  # Default
    if len(sys.argv) > 1:
        goal_subdir = sys.argv[1]
    
    print(f"Goal category: {goal_subdir}")
    print(f"Data root: {DATA_ROOT}")
    
    train_classifier(
        data_root=DATA_ROOT,
        goal_subdir=goal_subdir,
        model_size='large',
        img_dim=64,
        num_epochs=30,
        batch_size=64,
        lr=1e-3,
        val_split=0.2,
        max_neg_ratio=5.0,
    )


if __name__ == '__main__':
    main()
