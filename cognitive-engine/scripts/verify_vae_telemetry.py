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
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_H, IMG_W
from modules.spatial_model import TinyVAE

def load_latents_from_telemetry(vae_path, telemetry_df, device="cuda"):
    print(f"Loading VAE from {vae_path}...")
    state_dict = torch.load(vae_path, map_location=device)
    latent_dim, model_size, input_spatial_dim, _ = TinyVAE.detect_size(state_dict)
    
    print(f"Detected Dynamic Size: {model_size} | Latent: {latent_dim} | Input Spatial: {input_spatial_dim}")
    model = TinyVAE(model_size=model_size, latent_dim=latent_dim, input_spatial_dim=input_spatial_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])
    
    latents = []
    valid_indices = []
    
    print("Extracting latent space representations `z` for all telemetry frames...")
    
    with torch.no_grad():
        for idx, row in telemetry_df.iterrows():
            # In master_telemetry, ts is the frame identifier and img_dir wasn't explicitly saved
            # Wait! Did I save 'img_dir' or 'image_file' in master_telemetry.csv? 
            # In extract_telemetry.py I used ts, but didn't save base path!
            # Let's dynamically find it if it exists.
            
            # The exact logic from extract_telemetry generated ts from 'webcam_frame_XYZ.jpg'
            # Let's search the parent data directory for matching folders
            found = False
            base_dir = r"c:\Users\chris\Embrained\software_suite\data"
            
            # This is a brute force lookup for reliability if path isn't explicit
            for d in os.listdir(base_dir):
                if 'markov' in d:
                    ts_str = str(int(float(row['ts'])))
                    path = os.path.join(base_dir, d, 'images', f"webcam_frame_{ts_str}.jpg")
                    if os.path.exists(path):
                        img = Image.open(path).convert('RGB')
                        t_img = transform(img).unsqueeze(0).to(device)
                        _, mu, _ = model(t_img)
                        latents.append(mu.squeeze(0).cpu().numpy())
                        valid_indices.append(idx)
                        found = True
                        break
            if not found:
                print(f"Warning: Could not locate image for ts {row['ts']}")
                
    X = np.array(latents)
    telemetry_valid = telemetry_df.loc[valid_indices].reset_index(drop=True)
    return X, telemetry_valid

def evaluate_latents(vae_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    telemetry_path = "master_telemetry.csv"
    if not os.path.exists(telemetry_path):
        print("master_telemetry.csv not found! Run extract_telemetry.py first.")
        return
        
    df = pd.read_csv(telemetry_path)
    # Clean anomalies
    df = df[(df['ir'] > 0) & (df['ir'] < 4000) & (df['dist_px'] > 0)]
    
    X, df = load_latents_from_telemetry(vae_path, df, device)
    
    if len(X) == 0:
        print("No paired latents found! Check image paths.")
        return
        
    print(f"Successfully extracted {len(X)} Latent-to-Physics pairs.")
    
    targets = {
        'X Coordinate (px)': df['cx'].values,
        'Y Coordinate (px)': df['cy'].values,
        'Analog IR Reading': df['ir'].values,
        'Continuous Yaw (Cos)': np.cos(np.radians(df['yaw_deg'].values)), # Map angle to continuous space
        'Continuous Yaw (Sin)': np.sin(np.radians(df['yaw_deg'].values))
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"VAE Latent Space `z` Physical Grounding Validation\n{os.path.basename(vae_path)}", fontsize=18)
    axes = axes.flatten()
    
    results = {}
    
    for i, (name, y) in enumerate(targets.items()):
        print(f"\nTraining Regressor for {name}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use Random Forest to capture pure non-linear spatial mapping from latent topological dimensions
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = r2
        print(f"R^2 Score for {name}: {r2:.4f}")
        
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.3, color='blue')
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_title(f"{name}\nLatent R^2: {r2:.3f}", fontsize=14)
        ax.set_xlabel("True Explicit Physical Value")
        ax.set_ylabel("Predicted from 32D VAE Latent `z`")
        ax.grid(True)
        
    axes[5].axis('off') # Empty plot
    
    plt.tight_layout()
    out_img = f"latent_validation_{os.path.basename(vae_path)}.png"
    plt.savefig(out_img)
    print(f"\nSaved correlation array to {out_img}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VAE latents map directly to structural CV telemetry.")
    parser.add_argument("--vae_pth", type=str, required=True, help="Path to the trained VAE model.")
    args = parser.parse_args()
    
    evaluate_latents(args.vae_pth)
