import os
import sys
import time
import logging
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_H, IMG_W, DATA_DIR, MODELS_DIR
from modules.spatial_model import TinyVAE
from backend.services.datasets import DatasetService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainContinuousVAE")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VAEDataset(Dataset):
    def __init__(self, data_root):
        self.ds_service = DatasetService(data_root)
        self.samples = []
        datasets_info = self.ds_service.list_datasets(fast=True)
        for ds_info in datasets_info.get('datasets', []):
            ds_path = os.path.join(data_root, ds_info['name'])
            if not os.path.exists(os.path.join(ds_path, "episode_data.csv")): continue
            transitions = self.ds_service.load_transitions(ds_path)
            for t in transitions:
                if t.get('format') == 'markov':
                    t['image_path'] = os.path.join(ds_path, t['image_path'])
                self.samples.append(t)
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            if 'image_path' in sample and os.path.exists(sample['image_path']):
                img = Image.open(sample['image_path']).convert('RGB')
                return self.transform(img)
            else:
                raise ValueError("No image source found")
        except Exception as e:
            return torch.zeros((3, IMG_H, IMG_W))

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # MSE Loss
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # clamped to prevent extreme gradients early in training
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta-VAE Objective Function
    return MSE + beta * KLD, MSE, KLD

def main():
    parser = argparse.ArgumentParser(description="Train Continuous beta-VAE")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta-VAE KL penalty weight")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Epochs over which to scale beta from 0 to target")
    args = parser.parse_args()

    logger.info(f"Starting Continuous VAE Training on {DEVICE}")
    logger.info(f"Latent Dim: {args.latent_dim}, Beta: {args.beta}")
    
    dataset = VAEDataset(DATA_DIR)
    if len(dataset) == 0:
        logger.error("No valid dataset found in data directory. Aborting.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    logger.info(f"Dataset Size: {len(dataset)} frames")
    
    model = TinyVAE(
        latent_dim=args.latent_dim, 
        model_size='large', 
        input_spatial_dim=IMG_W, 
        in_channels=3
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Optional mixed precision
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_mse = 0
        total_kld = 0
        
        # Warmup Beta to prevent initial shock collapse
        current_beta = args.beta
        if epoch < args.warmup_epochs:
            current_beta = args.beta * (epoch / float(args.warmup_epochs))
            
        t0 = time.time()
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_idx, imgs in enumerate(iterator):
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    recon, mu, logvar = model(imgs)
                    loss, mse, kld = loss_function(recon, imgs, mu, logvar, beta=current_beta)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, mu, logvar = model(imgs)
                loss, mse, kld = loss_function(recon, imgs, mu, logvar, beta=current_beta)
                loss.backward()
                optimizer.step()
                
            loop_loss = loss.item()
            total_loss += loop_loss
            total_mse += mse.item()
            total_kld += kld.item()
            
            iterator.set_postfix(loss=loop_loss/args.batch_size)
                
        t1 = time.time()
        
        # Normalize metrics per sample
        samples = len(dataset)
        avg_loss = total_loss / samples
        avg_mse = total_mse / samples
        avg_kld = total_kld / samples
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} ({t1-t0:.1f}s) | Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | KLD: {avg_kld:.4f} | Beta: {current_beta:.3f}")

    # Save Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"tinyvae-vae_continuous_{timestamp}.pth"
    model_path = os.path.join(DATA_DIR, model_name)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Export Global Latents for CQL targeting
    try:
        from backend.train_vae import export_global_latents
        logger.info("Extracting global structural latents...")
        export_global_latents(model, DATA_DIR, model_name, architecture='continuous')
        logger.info("Global latent extraction complete.")
    except Exception as e:
        logger.error(f"Failed to export global latents: {e}")

if __name__ == "__main__":
    main()
