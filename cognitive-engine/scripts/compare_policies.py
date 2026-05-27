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
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from pathlib import Path

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Attempt to import CQLNetwork
try:
    from modules.spatial_model import CQLNetwork
    # We also need to define INPUT_DIM and HIDDEN_DIM if not importing config
    # Defaulting to values seen in train_cql.py/config.py
    INPUT_DIM = 64
    HIDDEN_DIM = 256
    ACTION_DIM = 5
except ImportError:
    print("Could not import CQLNetwork. Policy analysis will be skipped.")
    CQLNetwork = None

# Constants
ACTION_NAMES = {0: "FWD", 1: "LEFT", 2: "RIGHT", 3: "STOP", 4: "BACK"}
ACTIONS_LIST = ["FWD", "LEFT", "RIGHT", "STOP", "BACK"]

dataset_roots = {
    "1D": r"C:\Users\chris\Embrained\embrained-app\data\Nook",
    "2D": r"C:\Users\chris\Embrained\embrained-app\data\Livingroom"
}

def plot_to_b64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def analyze_policy(root):
    if CQLNetwork is None: return None, None

    name = os.path.basename(root)
    # Strict naming: [dataset]-vae-cql.pth
    model_path = os.path.join(root, f"{name}-vae-cql.pth")
    
    if not os.path.exists(model_path):
        print(f"No CQL model found for {name} (expected at {os.path.basename(model_path)})")
        return None, None

    # Load Model
    # Since we only need weights visualization, CPU is fine
    device = 'cpu' 
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Identify Final Layer Weights
        # CQLNetwork: fc4 is the output layer
        if 'fc4.weight' not in state_dict:
            print(f"Error: fc4.weight not found in {model_path}")
            return None, model_path
            
        weights = state_dict['fc4.weight'].numpy() # [ActionDim, HiddenDim]
        
        # Analysis Logic (from backend/training.py)
        # 1. For each hidden node (column), find the action (row) with the maximum weight
        max_indices = np.argmax(weights, axis=0)
        
        # 2. Count occurrences
        counts = [0] * ACTION_DIM
        for idx in max_indices:
            if idx < ACTION_DIM:
                counts[idx] += 1
                
        return counts, model_path
        
    except Exception as e:
        print(f"Failed to load policy {model_path}: {e}")
        return None, model_path

def generate_report():
    html_sections = []
    
    # --- HEADER ---
    html_sections.append("<html><head><style>body{font-family:sans-serif; padding:20px;} h2{border-bottom:1px solid #ccc;} .row{display:flex; gap:20px;} .col{flex:1;} img{max-width:100%;} table{border-collapse:collapse; width:100%;} th,td{border:1px solid #ddd; padding:8px; text-align:left;} th{background-color:#f2f2f2;}</style></head><body>")
    html_sections.append("<h1>Policy Comparison Report</h1>")
    
    comparison_data = []
    
    for name, root in dataset_roots.items():
        print(f"Processing {name}...")
        
        counts, model_path = analyze_policy(root)
        
        data = {
            "name": name,
            "counts": counts,
            "model_path": model_path
        }
        comparison_data.append(data)
        
    # --- TABLE: GENERAL STATS ---
    html_sections.append("<h2>Model Status</h2>")
    html_sections.append("<table><tr><th>Metric</th><th>SnowflakePrimary</th><th>Livingroom</th></tr>")
    
    d1 = comparison_data[0]
    d2 = comparison_data[1]
    
    html_sections.append(f"<tr><td>Dataset Path</td><td>{dataset_roots['1D']}</td><td>{dataset_roots['2D']}</td></tr>")
    html_sections.append(f"<tr><td>CQL Model</td><td>{os.path.basename(d1['model_path']) if d1['model_path'] else 'N/A'}</td><td>{os.path.basename(d2['model_path']) if d2['model_path'] else 'N/A'}</td></tr>")
    html_sections.append("</table>")
    
    # --- PLOTS: ACTION DISTRIBUTION ---
    html_sections.append("<h2>Final Layer Action Preference</h2>")
    html_sections.append("<p>This plot shows how many neurons in the final hidden layer project most strongly to each action. A balanced distribution typically indicates a healthy policy.</p>")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, d in enumerate(comparison_data):
        if d['counts']:
            axes[i].bar(ACTIONS_LIST, d['counts'], color='#6366f1')
            axes[i].set_title(f"{d['name']} Policy Distribution")
            axes[i].set_ylabel("Node Count")
            
            # Annotate bars
            for j, v in enumerate(d['counts']):
                axes[i].text(j, v + 1, str(v), ha='center')
        else:
            axes[i].text(0.5, 0.5, "No Model / Load Error", ha='center')
            axes[i].set_title(f"{d['name']} (Missing)")
    
    plt.tight_layout()
    html_sections.append(f'<img src="data:image/png;base64,{plot_to_b64()}"/>')
        
    html_sections.append("</body></html>")
    
    with open("policy_comparison.html", "w") as f:
        f.write("\n".join(html_sections))
        
    print("Report generated: policy_comparison.html")

if __name__ == "__main__":
    generate_report()
