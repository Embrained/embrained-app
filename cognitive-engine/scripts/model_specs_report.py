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


import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.spatial_model import TinyVAE, CQLNetwork

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_vae_specs(model_size):
    try:
        model = TinyVAE(latent_dim=32, model_size=model_size)
    except Exception as e:
        return f"Error: {e}", {}, 0

    base_channels = model.base_channels
    param_count = count_parameters(model)
    
    # Introspect Encoder layers
    layers_desc = []
    # Simplified introspection based on code knowledge
    # We can infer depth from the loop in the code or check the module list
    
    # Count Conv2d layers in encoder
    conv_count = sum(1 for m in model.encoder if isinstance(m, nn.Conv2d))
    
    flatten_dim = model.flatten_dim
    
    specs = {
        "Base Channels": base_channels,
        "Conv Layers (Encoder)": conv_count,
        "Flatten Dim": flatten_dim,
        "Latent Dim": model.latent_dim
    }
    return specs, param_count

def get_cql_specs(model_size):
    try:
        model = CQLNetwork(input_dim=64, hidden_dim=256, action_dim=4, model_size=model_size)
    except Exception as e:
        return f"Error: {e}", {}, 0
        
    param_count = count_parameters(model)
    
    # Introspect
    # Check hidden layers list
    hidden_layers_count = len(model.hidden_layers)
    total_layers = hidden_layers_count + 2 # input + output
    
    # Get hidden dim from input layer output
    hidden_dim = model.input_layer.out_features
    
    specs = {
        "Hidden Dim": hidden_dim,
        "Total Layers": total_layers, # Including input/output
        "Activation": "ReLU",
        "LayerNorm": "Yes" if model.use_ln else "No"
    }
    return specs, param_count

def generate_report():
    vae_sizes = ['tiny', 'small', 'medium', 'large', 'enormous']
    cql_sizes = ['small', 'medium', 'large', 'enormous'] # 'tiny' maps to 'small' in code logic actually, but let's check explicit behavior
    
    html = """
    <html>
    <head>
        <title>Embrained Model Specifications</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; color: #e0e0e0; padding: 20px; }
            h1, h2 { color: #4facfe; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
            th, td { text-align: left; padding: 12px; border-bottom: 1px solid #333; }
            th { background-color: #2d2d2d; color: #ffa726; }
            tr:hover { background-color: #2a2a2a; }
            .spec-val { color: #81c784; font-family: monospace; }
        </style>
    </head>
    <body>
        <h1>Model Specifications</h1>
        
        <h2>Vision System (TinyVAE)</h2>
        <p>Input: 64x64 RGB Image. Latent Dim: 32.</p>
        <table>
            <tr>
                <th>Size</th>
                <th>Base Channels</th>
                <th>Conv Layers</th>
                <th>Flatten Dim</th>
                <th>Parameters</th>
            </tr>
    """
    
    for size in vae_sizes:
        specs, params = get_vae_specs(size)
        if isinstance(specs, str):
            html += f"<tr><td>{size.title()}</td><td colspan='4'>{specs}</td></tr>"
            continue
            
        html += f"""
            <tr>
                <td>{size.title()}</td>
                <td class='spec-val'>{specs['Base Channels']}</td>
                <td class='spec-val'>{specs['Conv Layers (Encoder)']}</td>
                <td class='spec-val'>{specs['Flatten Dim']}</td>
                <td class='spec-val'>{params:,}</td>
            </tr>
        """
        
    html += """
        </table>
        
        <h2>Control Policy (CQLNetwork)</h2>
        <p>Input: 64 Dim (Stacked Latents). Output: 4 Actions.</p>
        <table>
            <tr>
                <th>Size</th>
                <th>Hidden Dim</th>
                <th>Hidden Layers</th>
                <th>Total Depth</th>
                <th>Parameters</th>
            </tr>
    """

    for size in cql_sizes:
        specs, params = get_cql_specs(size)
        if isinstance(specs, str):
            html += f"<tr><td>{size.title()}</td><td colspan='4'>{specs}</td></tr>"
            continue
            
        html += f"""
            <tr>
                <td>{size.title()}</td>
                <td class='spec-val'>{specs['Hidden Dim']}</td>
                <td class='spec-val'>{specs['Total Layers'] - 2}</td>
                <td class='spec-val'>{specs['Total Layers']}</td>
                <td class='spec-val'>{params:,}</td>
            </tr>
        """
        
    html += """
        </table>
        <p><small>Generated by model_specs_report.py</small></p>
    </body>
    </html>
    """
    
    with open("model_specs.html", "w") as f:
        f.write(html)
    
    print("Report generated: model_specs.html")

if __name__ == "__main__":
    generate_report()
