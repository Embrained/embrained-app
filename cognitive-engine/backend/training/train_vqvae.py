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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import IMG_H, IMG_W, DATA_DIR, MODELS_DIR
from modules.spatial_model import DiscreteVQVAE
from backend.services.datasets import DatasetService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainVQVAE")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VAEDataset(Dataset):
    def __init__(self, data_root):
        import json
        self.ds_service = DatasetService(data_root)
        self.samples = []
        self.data_root = data_root
        
        json_path = os.path.join(data_root, "all_transitions.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                all_data = json.load(f)
        else:
            raise FileNotFoundError(f"Could not find curated dataset at {json_path}")
            
        for item in all_data:
            p = item.get('image_path', '')
            if 'webcam_frame_' not in p and 'frame_' in os.path.basename(p):
                # Ensure path is absolute or relative to data_root
                if not os.path.isabs(p):
                    p = os.path.join(self.data_root, p)
                if os.path.exists(p):
                    item['absolute_image_path'] = p
                    self.samples.append(item)
                    
        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample['absolute_image_path']).convert('RGB')
            return self.transform(img)
        except Exception as e:
            return torch.zeros((3, IMG_H, IMG_W))

def main():
    parser = argparse.ArgumentParser(description="Train Discrete VQ-VAE")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (VQ-VAEs prefer slightly higher LRs)") 
    parser.add_argument("--latent_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Codebook size")
    args = parser.parse_args()

    logger.info(f"Starting DiscreteVQVAE Training on {DEVICE}")
    logger.info(f"Codebook Size: {args.num_embeddings}, Embed Dim: {args.latent_dim}")
    
    dataset = VAEDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    logger.info(f"Dataset Size: {len(dataset)} frames")
    
    model = DiscreteVQVAE(
        latent_dim=args.latent_dim, 
        model_size='large', 
        input_spatial_dim=IMG_W, 
        in_channels=3, 
        num_embeddings=args.num_embeddings
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Optional mixed precision
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_recon = 0
        total_vq = 0
        total_perp = 0
        
        t0 = time.time()
        for batch_idx, imgs in enumerate(dataloader):
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    recon, quantized, vq_loss, perplexity = model(imgs)
                    recon_loss = nn.functional.mse_loss(recon, imgs)
                    loss = recon_loss + vq_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, quantized, vq_loss, perplexity = model(imgs)
                recon_loss = nn.functional.mse_loss(recon, imgs)
                loss = recon_loss + vq_loss
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_perp += perplexity.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}", end='\r')
                
        t1 = time.time()
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_vq = total_vq / len(dataloader)
        avg_perp = total_perp / len(dataloader)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} ({t1-t0:.1f}s) | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} | Perplexity: {avg_perp:.2f}")

    # Save Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"vqvae_{args.num_embeddings}c_{args.latent_dim}d_{timestamp}.pth"
    model_path = os.path.join(DATA_DIR, model_name)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
