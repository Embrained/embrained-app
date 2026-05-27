import os
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.spatial_model import TinyVAE

def probe_latent_space():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Initiating Latent Geometry Probe...")
    
    # 1. Load Telemetry
    telemetry = {}
    master_path = os.path.join(data_root, "master_teleaiser.csv") # Try fallback
    if not os.path.exists(master_path):
        master_path = os.path.join(data_root, 'master_telemetry.csv')
    df = pd.read_csv(master_path)
    for _, row in df.iterrows():
        try:
            yaw_rad = math.radians(row['yaw_deg'])
            telemetry[str(row['ts'])] = [
                float(row['cx']),
                float(row['cy']),
                math.cos(yaw_rad),
                math.sin(yaw_rad)
            ]
        except KeyError:
            continue
            
    # 2. Get latest VAE Cache (already extracted latents!)
    import glob
    vae_candidates = glob.glob(os.path.join(data_root, '*-vae_*.pth'))
    if not vae_candidates:
        print("No VAE found!")
        return
        
    vae_candidates.sort(key=os.path.getmtime, reverse=True)
    target_vae = vae_candidates[0]
    vae_basename = os.path.basename(target_vae).replace('.pth', '')
    cache_path = os.path.join(data_root, f"{vae_basename}_global_latents.pt")
    
    if not os.path.exists(cache_path):
        print("No latent cache found, please run standard evaluate_oracles.py first.")
        return
        
    print(f"Loading Latent Cache: {cache_path}")
    latent_dict = torch.load(cache_path, map_location='cpu', weights_only=True).get("ts_map", {})
    
    # 3. Join logic
    latents = []
    targets = []
    
    for ts, l in latent_dict.items():
        if ts in telemetry:
            latents.append(l.numpy())
            targets.append(telemetry[ts])
            
    if not latents:
        print("Intersection empty!")
        return
        
    X = np.stack(latents)
    y = np.array(targets)
    
    print(f"Dataset compiled: {X.shape[0]} samples. Feature dim: {X.shape[1]}")
    
    # 4. Fit Linear Probes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We probe purely with linear regression. If linear regression works, the geometry is linearly decodable!
    model_xy = Ridge(alpha=1.0)
    model_xy.fit(X_train, y_train[:, :2]) # predict cx, cy
    
    model_yaw = Ridge(alpha=1.0)
    model_yaw.fit(X_train, y_train[:, 2:]) # predict cos, sin
    
    pred_xy = model_xy.predict(X_test)
    pred_yaw = model_yaw.predict(X_test)
    
    r2_xy = r2_score(y_test[:, :2], pred_xy)
    r2_yaw = r2_score(y_test[:, 2:], pred_yaw)
    
    print(f"Linear R^2 for XY Position: {r2_xy:.4f}")
    print(f"Linear R^2 for Yaw (cos/sin): {r2_yaw:.4f}")
    
    # 5. Visualizer
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], y_test[:, 1], c='blue', alpha=0.5, label='True Position', s=10)
    plt.scatter(pred_xy[:, 0], pred_xy[:, 1], c='red', alpha=0.5, label='Linear Decoded', s=10)
    plt.title(f"Decoded XY Space (R2={r2_xy:.2f})")
    plt.legend()
    plt.gca().invert_yaxis() # Camera coords
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 2], y_test[:, 3], c='blue', alpha=0.5, label='True Yaw Vector', s=10)
    plt.scatter(pred_yaw[:, 0], pred_yaw[:, 1], c='red', alpha=0.5, label='Linear Decoded Yaw', s=10)
    plt.title(f"Decoded Yaw Space (R2={r2_yaw:.2f})")
    plt.legend()
    
    out_path = os.path.join(data_root, 'latent_probe_results.png')
    plt.savefig(out_path)
    print(f"Probe complete. Chart saved to {out_path}")

if __name__ == "__main__":
    probe_latent_space()
