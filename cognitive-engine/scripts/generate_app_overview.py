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
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.spatial_model import TinyVAE, CQLNetwork
from config import IMG_H, IMG_W, INPUT_DIM, HIDDEN_DIM

def generate_report():
    html = []
    
    # Header & Style
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embrained App Overview</title>
    <style>
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 40px 20px; background-color: #fcfcfc; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; margin-bottom: 30px; }
        h2 { color: #2980b9; margin-top: 40px; margin-bottom: 15px; font-weight: 600; }
        h3 { color: #7f8c8d; margin-top: 25px; margin-bottom: 10px; font-weight: 600; }
        p { margin-bottom: 15px; }
        code { background: #f0f2f5; padding: 2px 6px; border-radius: 4px; font-family: Consolas, monospace; color: #d63384; font-size: 0.9em; }
        pre { background: #282c34; color: #abb2bf; padding: 20px; border-radius: 8px; overflow-x: auto; font-family: Consolas, monospace; font-size: 0.9em; line-height: 1.5; }
        ul, ol { padding-left: 25px; margin-bottom: 20px; }
        li { margin-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; margin: 25px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); background: white; border-radius: 8px; overflow: hidden; }
        th, td { text-align: left; padding: 15px; border-bottom: 1px solid #eee; }
        th { background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }
        tr:last-child td { border-bottom: none; }
        .card { background: white; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; background-color: #e3f2fd; color: #0d47a1; }
        .workflow-step { display: flex; align-items: flex-start; margin-bottom: 15px; }
        .step-number { background: #2980b9; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px; flex-shrink: 0; }
        .step-content { flex: 1; }
    </style>
</head>
<body>
    <h1>Embrained App Overview</h1>""")

    # 1. App Summary
    html.append("""
    <div class="card">
        <h2>1. Summary</h2>
        <p>The <strong>Embrained App</strong> is a robotic control system that uses a <strong>VAE (Variational Autoencoder)</strong> for vision and <strong>CQL (Conservative Q-Learning)</strong> for decision making. 
        It allows robots to learn latent representations of their environment and navigate towards visual goals. The system is designed to be hardware-agnostic, currently supporting:</p>
        <ul>
            <li><strong>Plexus</strong></li>
            <li><strong>SpikerBot</strong></li>
        </ul>
    </div>""")
    
    # 2. Features
    html.append("""
    <h2>2. Key Features</h2>
    <ul>
        <li><strong>Live Mode</strong>: Real-time inference and control of the robot using trained models.</li>
        <li><strong>Training Mode</strong>: Interface for training VAE (Vision) and CQL (Policy) models on collected datasets.</li>
        <li><strong>Manifold Visualization</strong>: 2D visualization of the high-dimensional latent space to understand model learning.</li>
        <li><strong>Data Collection</strong>: Tools to record datasets (camera feed + actions) for training.</li>
        <li><strong>Hardware Abstraction</strong>: Unified API for multiple robot embodiments (Plexus, SpikerBot).</li>
        <li><strong>Goal-Conditioned Policies</strong>: Robots navigate to specific visual target images specified by the user.</li>
    </ul>""")
        
    # 3. Data Flow Workflow
    html.append("""
    <h2>3. Data Flow Workflow</h2>
    <p>The following workflow describes how information flows through the system during autonomous navigation:</p>
    
    <div class="workflow-step">
        <div class="step-number">1</div>
        <div class="step-content"><strong>Input</strong>: Camera captures an RGB Image (64x64).</div>
    </div>
    <div class="workflow-step">
        <div class="step-number">2</div>
        <div class="step-content"><strong>Vision Processing (VAE Encoder)</strong>: The image is passed through the <code>TinyVAE</code> Encoder, compressing the high-dimensional input into a low-dimensional <strong>Latent Vector</strong> (<code>z_curr</code>, size 32).</div>
    </div>
    <div class="workflow-step">
        <div class="step-number">3</div>
        <div class="step-content"><strong>Goal Specification</strong>: The user selects a Goal Image, which is also encoded into a <strong>Goal Latent Vector</strong> (<code>z_goal</code>, size 32).</div>
    </div>
    <div class="workflow-step">
        <div class="step-number">4</div>
        <div class="step-content"><strong>State Construction</strong>: <code>z_curr</code> and <code>z_goal</code> are concatenated to form the system <strong>State Vector</strong> (size 64).</div>
    </div>
    <div class="workflow-step">
        <div class="step-number">5</div>
        <div class="step-content"><strong>Decision Making (CQL Policy)</strong>: The State Vector is passed to the <code>CQLNetwork</code>. The network outputs Q-values for 5 discrete actions (FWD, LEFT, RIGHT, STOP, BACK). The action with the highest Q-value is selected.</div>
    </div>
    <div class="workflow-step">
        <div class="step-number">6</div>
        <div class="step-content"><strong>Actuation</strong>: The selected Action ID is sent to the Hardware Interface or Simulator to execute the move.</div>
    </div>""")
    
    # 4. Model Architectures & Parameters
    html.append("<h2>4. Model Architectures & Parameters</h2>")
    
    # Instantiate models to inspect
    # (In a real scenario we'd use these objects to populate the text dynamically, but for this overview we can describe them)
    vae = TinyVAE(latent_dim=32)
    cql = CQLNetwork(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=5)
    
    html.append("""
    <div class="card">
        <h3>A. Vision Model: TinyVAE</h3>
        <p>A small Variational Autoencoder designed for 64x64 input, optimized for consumer hardware.</p>
        <p><strong>Architecture Details:</strong></p>
        <ul>
            <li><strong>Input</strong>: 64x64 x 3 (RGB)</li>
            <li><strong>Latent Dimension</strong>: 32</li>
            <li><strong>Encoder Layers</strong>:
                <ul>
                    <li>Conv2d(3 &rarr; 32, k=4, s=2, p=1) + ReLU</li>
                    <li>Conv2d(32 &rarr; 64, k=4, s=2, p=1) + ReLU</li>
                    <li>Conv2d(64 &rarr; 128, k=4, s=2, p=1) + ReLU</li>
                    <li>Flatten &rarr; Linear(8192 &rarr; 32) [Mu, LogVar]</li>
                </ul>
            </li>
            <li><strong>Decoder Layers</strong>:
                <ul>
                    <li>Linear(32 &rarr; 8192) &rarr; Unflatten</li>
                    <li>ConvTranspose2d(128 &rarr; 64) + ReLU</li>
                    <li>ConvTranspose2d(64 &rarr; 32) + ReLU</li>
                    <li>ConvTranspose2d(32 &rarr; 3) + Sigmoid</li>
                </ul>
            </li>
        </ul>
    </div>""")

    html.append(f"""
    <div class="card">
        <h3>B. Policy Model: CQLNetwork</h3>
        <p>A Multi-Layer Perceptron (MLP) trained with Conservative Q-Learning.</p>
        <p><strong>Architecture Details:</strong></p>
        <ul>
            <li><strong>Input Dimension</strong>: {INPUT_DIM} (32 current + 32 goal)</li>
            <li><strong>Hidden Dimension</strong>: {HIDDEN_DIM}</li>
            <li><strong>Structure</strong>:
                <ul>
                    <li>LayerNorm(Input)</li>
                    <li>Linear(Input &rarr; Hidden) + ReLU</li>
                    <li>Linear(Hidden &rarr; Hidden) + ReLU</li>
                    <li>Linear(Hidden &rarr; Action Dim (5))</li>
                </ul>
            </li>
        </ul>
    </div>""")
    
    # 5. Training Parameters (Datasets)
    html.append("""
    <h2>5. Dataset Configuration & Training Parameters</h2>
    
    <h3>Common Parameters</h3>
    <table>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Source</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>VAE Learning Rate</td><td>1e-4</td><td><code>train_vae.py</code></td></tr>
            <tr><td>VAE Batch Size</td><td>64</td><td><code>train_vae.py</code></td></tr>
            <tr><td>VAE Beta (Disentangle)</td><td>4.0</td><td><code>train_vae.py</code></td></tr>
            <tr><td>CQL Learning Rate</td><td>1e-4</td><td><code>train_cql.py</code></td></tr>
            <tr><td>CQL Gamma</td><td>0.99</td><td><code>train_cql.py</code></td></tr>
            <tr><td>CQL Alpha</td><td>5.0</td><td><code>train_cql.py</code></td></tr>
        </tbody>
    </table>

    <h3>Specific Datasets</h3>
    
    <h4>1. Nook Dataset (1D)</h4>
    <ul>
        <li><strong>Description</strong>: Linear track environment.</li>
        <li><strong>Typical Epochs</strong>: 10 (Default)</li>
        <li><strong>Structure</strong>: Linear 1D Manifold (Latent space forms a ring/line).</li>
    </ul>
    
    <h4>2. Livingroom Dataset (2D)</h4>
    <ul>
        <li><strong>Description</strong>: Open 2D floor environment.</li>
        <li><strong>Typical Epochs</strong>: 10 (Default)</li>
        <li><strong>Structure</strong>: 2D Manifold (Latent space forms a scattered map/blob).</li>
    </ul>""")
    
    html.append("</body></html>")
    
    output_path = os.path.join(os.path.dirname(__file__), "../app_overview.html")
    with open(output_path, "w") as f:
        f.write("\n".join(html))
        
    print(f"Overview generated at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_report()
