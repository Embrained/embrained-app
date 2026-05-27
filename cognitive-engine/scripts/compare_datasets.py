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
import json
import sys
import base64
import io
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from pathlib import Path
from PIL import Image

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Attempt to import TinyVAE
try:
    from modules.spatial_model import TinyVAE
    # helper for transform
    from torchvision import transforms
    IMG_H, IMG_W = 64, 64
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])
except ImportError:
    print("Could not import TinyVAE. Latent analysis will be skipped.")
    TinyVAE = None

# Constants
ACTION_NAMES = {0: "FWD", 1: "LEFT", 2: "RIGHT", 3: "STOP", 4: "BACK"}
dataset_roots = {
    "1D": r"C:\Users\chris\Embrained\embrained-app\data\Nook",
    "2D": r"C:\Users\chris\Embrained\embrained-app\data\Livingroom"
}

def discretize_action(left, right):
    tol = 1
    if abs(left) < 1 and abs(right) < 1: return 3
    if left < -tol and right > tol: return 0
    if left < -tol and right < -tol: return 1
    if left > tol and right > tol: return 2
    if left > tol and right < -tol: return 4
    return 3

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)

def compute_action_dist(transitions):
    counts = {k: 0 for k in ACTION_NAMES.keys()}
    sequence = []
    
    for t in transitions:
        l = t.get('left_cmd', 0)
        r = t.get('right_cmd', 0)
        a = discretize_action(l, r)
        counts[a] += 1
        sequence.append(a)
        
    return counts, sequence

def compute_markov(sequence):
    # Transition matrix
    n_actions = len(ACTION_NAMES)
    matrix = np.zeros((n_actions, n_actions))
    
    for i in range(len(sequence)-1):
        curr = sequence[i]
        next_a = sequence[i+1]
        matrix[curr][next_a] += 1
        
    # Normalize
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_matrix = np.divide(matrix, row_sums)
        prob_matrix = np.nan_to_num(prob_matrix)
        
    return prob_matrix

def compute_latents(root, transitions, sample_size=500):
    if TinyVAE is None: return None, None, None
    
    # 1. Look for VAE model in root
    name = os.path.basename(root)
    # Strict naming per user request: [dataset]-vae.pth
    target_path = os.path.join(root, f"{name}-vae.pth")
    
    if os.path.exists(target_path):
        model_path = target_path
    else:
        print(f"No VAE found for {name} (expected locally at {os.path.basename(target_path)})")
        return None, None, None

    # [NEW] Check for cached manifold.pkl (contains PCA object)
    manifold_path = os.path.join(root, "manifold.pkl")
    pca_model = None
    
    if os.path.exists(manifold_path):
        try:
            with open(manifold_path, 'rb') as f:
                data = pickle.load(f)
                if 'pca' in data:
                    pca_model = data['pca']
                    print(f"Loaded cached PCA from {manifold_path}")
        except Exception as e:
            print(f"Failed to load manifold cache: {e}")

    # Load VAE Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyVAE(latent_dim=32).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    try:
        if 'encoder.0.weight' in state_dict or 'encoder.0.0.weight' in state_dict:
            model.load_state_dict(state_dict, strict=False)
        else:
             model.encoder.load_state_dict(state_dict)
    except:
        try: model.load_state_dict(state_dict, strict=False)
        except: pass
        
    model.eval()
    
    # Sample images
    # If using cached PCA, we might want consistent latents?
    # But usually we want to see where *this* set of transitions projects.
    # So we re-compute latents for the sampled transitions, but use the SAME PCA to project them.
    
    samples = random.sample(transitions, min(len(transitions), sample_size))
    latents = []
    
    with torch.no_grad():
        for s in samples:
            path = os.path.join(root, s['image_path'])
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    t_img = transform(img).unsqueeze(0).to(device)
                    # Get Mu
                    _, mu, _ = model(t_img)
                    latents.append(mu.cpu().numpy().flatten())
                    
    return np.array(latents), model_path, pca_model

def plot_to_b64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_report():
    html_sections = []
    
    # --- HEADER ---
    html_sections.append("<html><head><style>body{font-family:sans-serif; padding:20px;} h2{border-bottom:1px solid #ccc;} .row{display:flex; gap:20px;} .col{flex:1;} img{max-width:100%;} table{border-collapse:collapse; width:100%;} th,td{border:1px solid #ddd; padding:8px; text-align:left;} th{background-color:#f2f2f2;}</style></head><body>")
    html_sections.append("<h1>Dataset Comparison Report</h1>")
    
    comparison_data = []
    
    for name, root in dataset_roots.items():
        print(f"Processing {name}...")
        trans_path = os.path.join(root, "all_transitions.json")
        eps_path = os.path.join(root, "episodes.json")
        
        trans = load_json(trans_path)
        eps = load_json(eps_path)
        
        counts, sequence = compute_action_dist(trans)
        markov = compute_markov(sequence)
        
        latents, vpath, pca_model = compute_latents(root, trans)
        
        # Check for CQL model
        # Strict naming: [dataset]-vae-cql.pth
        cql_path = os.path.join(root, f"{name}-vae-cql.pth")
        if not os.path.exists(cql_path):
            cql_path = None
        
        data = {
            "name": name,
            "n_transitions": len(trans),
            "n_episodes": len(eps),
            "counts": counts,
            "sequence": sequence,
            "markov": markov,
            "latents": latents,
            "vae_path": vpath,
            "cql_path": cql_path,
            "pca_model": pca_model 
        }
        comparison_data.append(data)
        
    # --- TABLE: GENERAL STATS ---
    html_sections.append("<h2>General Statistics</h2>")
    html_sections.append("<table><tr><th>Metric</th><th>SnowflakePrimary</th><th>Livingroom</th></tr>")
    
    d1 = comparison_data[0]
    d2 = comparison_data[1]
    
    html_sections.append(f"<tr><td>Transitions</td><td>{d1['n_transitions']}</td><td>{d2['n_transitions']}</td></tr>")
    html_sections.append(f"<tr><td>Episodes</td><td>{d1['n_episodes']}</td><td>{d2['n_episodes']}</td></tr>")
    html_sections.append(f"<tr><td>VAE Model</td><td>{os.path.basename(d1['vae_path']) if d1['vae_path'] else 'N/A'}</td><td>{os.path.basename(d2['vae_path']) if d2['vae_path'] else 'N/A'}</td></tr>")
    html_sections.append(f"<tr><td>CQL Model</td><td>{os.path.basename(d1['cql_path']) if d1['cql_path'] else 'N/A'}</td><td>{os.path.basename(d2['cql_path']) if d2['cql_path'] else 'N/A'}</td></tr>")
    html_sections.append("</table>")
    
    # --- PLOTS: ACTION DISTRIBUTION ---
    html_sections.append("<h2>Action Distribution</h2>")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, d in enumerate(comparison_data):
        acts = list(d['counts'].keys())
        freqs = list(d['counts'].values())
        total = sum(freqs) if sum(freqs) > 0 else 1
        pcts = [f/total*100 for f in freqs]
        labels = [ACTION_NAMES[a] for a in acts]
        
        axes[i].bar(labels, pcts, color='teal')
        axes[i].set_title(f"{d['name']} Actions")
        axes[i].set_ylabel("Frequency (%)")
        axes[i].set_ylim(0, 100)
    
    html_sections.append(f'<img src="data:image/png;base64,{plot_to_b64()}"/>')
    
    # --- PLOTS: MARKOV CHAINS ---
    html_sections.append("<h2>Markov Transition Probabilities</h2>")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    act_labels = [ACTION_NAMES[i] for i in range(5)]
    
    for i, d in enumerate(comparison_data):
        cax = axes[i].matshow(d['markov'], cmap="Blues", vmin=0, vmax=1)
        axes[i].set_xticks(range(len(act_labels)))
        axes[i].set_yticks(range(len(act_labels)))
        axes[i].set_xticklabels(act_labels)
        axes[i].set_yticklabels(act_labels)
        axes[i].set_title(f"{d['name']} Transition Probabilities")
        
        # Annotate
        for (j, k), z in np.ndenumerate(d['markov']):
            axes[i].text(k, j, f'{z:.2f}', ha='center', va='center', color='black' if z < 0.5 else 'white')

    html_sections.append(f'<img src="data:image/png;base64,{plot_to_b64()}"/>')
    
    # --- PLOTS: LATENT SPACE (PCA) ---
    html_sections.append("<h2>Latent Space Structure (PCA)</h2>")
    if d1['latents'] is not None and d2['latents'] is not None:
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, d in enumerate(comparison_data):
            if d['latents'] is not None and len(d['latents']) > 2:
                # [FIX] Use Cached PCA if available
                if d['pca_model']:
                    print(f"Using cached PCA for {d['name']}")
                    lpoints = d['pca_model'].transform(d['latents'])
                    title_suffix = "(Cached PCA)"
                else:
                    print(f"Fitting New PCA for {d['name']}")
                    pca = PCA(n_components=2)
                    lpoints = pca.fit_transform(d['latents'])
                    title_suffix = "(New PCA)"
                
                # Scatter
                axes[i].scatter(lpoints[:,0], lpoints[:,1], alpha=0.5, s=10, c='purple')
                axes[i].set_title(f"{d['name']} Latent Manifold {title_suffix}")
                axes[i].set_xlabel("PC1")
                axes[i].set_ylabel("PC2")
                
                # Variance Ratio (Only if we fitted it or can retrieve it)
                if d['pca_model']:
                    var = d['pca_model'].explained_variance_ratio_
                else:
                    var = pca.explained_variance_ratio_
                    
                axes[i].text(0.05, 0.95, f"Var: {var[0]:.2f}, {var[1]:.2f}", 
                             transform=axes[i].transAxes, verticalalignment='top')
            else:
                axes[i].text(0.5, 0.5, "Insufficient Data/No Model", ha='center')
                
        html_sections.append(f'<img src="data:image/png;base64,{plot_to_b64()}"/>')
    else:
        html_sections.append("<p>Could not generate latent comparison (Models missing)</p>")
        
    html_sections.append("</body></html>")
    
    with open("dataset_comparison.html", "w") as f:
        f.write("\n".join(html_sections))
        
    print("Report generated: dataset_comparison.html")

if __name__ == "__main__":
    generate_report()
