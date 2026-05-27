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

import torch
import torch.nn.functional as F
import json
import numpy as np
import sys
import os

# Add root to sys path
sys.path.append(r"c:\Users\chris\Embrained\software_suite")

from backend.models.latentslam import LatentSLAM
from backend.training.datasets.latentslam_dataset import LatentSLAMDataset

def evaluate_forward_prediction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    import glob
    model_pattern = r"c:\Users\chris\Embrained\software_suite\data\latentslam_*.pth"
    models = glob.glob(model_pattern)
    if not models:
        print(f"No models matching {model_pattern} found.")
        return
    model_path = max(models, key=os.path.getctime)
    print(f"Loading latest model: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    hidden_dim = 256
    latent_dim = 128
    if 'posterior_fc.0.weight' in state_dict:
        hidden_dim = state_dict['posterior_fc.0.weight'].shape[0]
        latent_dim = state_dict['posterior_fc.2.weight'].shape[0] // 2
        
    model = LatentSLAM(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load dataset
    data_root = r"c:\Users\chris\Embrained\software_suite\data"
    all_transitions_path = os.path.join(data_root, "all_transitions.json")
    with open(all_transitions_path, 'r') as f:
        transitions = json.load(f)
        
    dataset = LatentSLAMDataset(transitions, data_root=data_root, device=device)
    
    # Filter for macro_action == 1 (Forward)
    forward_samples = []
    for sample in dataset.samples:
        if sample['curr_node'].get('macro_action') == 1:
            forward_samples.append(sample)
            
    print(f"Found {len(forward_samples)} forward macro_action samples out of {len(dataset.samples)}")
    
    if len(forward_samples) == 0:
        print("No forward samples found. Check macro_action == 1 condition.")
        return
        
    mses = []
    eps_dist = []
    all_z_t = []
    
    with torch.no_grad():
        for i, sample in enumerate(forward_samples):
            img_curr = dataset._load_img(sample['curr_node']).unsqueeze(0).to(device)
            img_next = dataset._load_img(sample['next_node']).unsqueeze(0).to(device)
            
            action = torch.tensor(sample['action'], dtype=torch.float32).unsqueeze(0).to(device)
            action = action / 255.0  # Normalize PWM
            
            # z_t
            b = img_curr.size(0)
            z_init_prior = torch.zeros(b, model.latent_dim, device=device)
            a_init_prior = torch.zeros(b, model.action_dim, device=device)
            
            # Encode current state
            z_t, _ = model.get_posterior(z_init_prior, a_init_prior, img_curr)
            all_z_t.append(z_t)
            
            # Ground-truth z_{t+1}
            z_next, _ = model.get_posterior(z_t, action, img_next)
            
            # Hallucinate \hat{z}_{t+1}
            z_hat, _ = model.get_prior(z_t, action)
            
            mse = F.mse_loss(z_hat, z_next).item()
            sq_dist = torch.sum((z_hat - z_next)**2).item()
            euclid_dist = np.sqrt(sq_dist)
            
            mses.append(mse)
            eps_dist.append(euclid_dist)
            
    avg_mse = np.mean(mses)
    avg_dist = np.mean(eps_dist)
    
    all_z_t_tensor = torch.cat(all_z_t, dim=0)
    variances = torch.var(all_z_t_tensor, dim=0)
    
    print("=== Eval Results ===")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average Euclidean Dist: {avg_dist:.4f}")
    print(f"Min Dist: {np.min(eps_dist):.4f}, Max Dist: {np.max(eps_dist):.4f}")
    print(f"Max Variance of z_t (across dims): {variances.max().item():.6f}")
    print(f"Mean Variance of z_t (across dims): {variances.mean().item():.6f}")

    # --- VISUAL SANITY CHECK ---
    print("\nGenerating 5 Visual Sanity Checks...")
    import matplotlib.pyplot as plt
    try:
        for i in range(5):
            # Pick a random sample from the forward samples
            sample_idx = np.random.randint(0, len(forward_samples))
            sample = forward_samples[sample_idx]
            
            img_curr = dataset._load_img(sample['curr_node']).unsqueeze(0).to(device)
            img_next = dataset._load_img(sample['next_node']).unsqueeze(0).to(device)
            action = torch.tensor(sample['action'], dtype=torch.float32).unsqueeze(0).to(device) / 255.0
            
            b = img_curr.size(0)
            z_init_prior = torch.zeros(b, model.latent_dim, device=device)
            a_init_prior = torch.zeros(b, model.action_dim, device=device)
            
            # 1. Encode t
            z_t, _ = model.get_posterior(z_init_prior, a_init_prior, img_curr)
            recon_t_tensor = model.get_likelihood(z_t)
            
            # 2. Encode t+1 (Ground Truth)
            z_next, _ = model.get_posterior(z_t, action, img_next)
            recon_next_tensor = model.get_likelihood(z_next)
            
            # 3. Predict t+1 (Hallucination)
            z_hat, _ = model.get_prior(z_t, action)
            recon_hat_tensor = model.get_likelihood(z_hat)
            
            def to_numpy_img(tensor):
                return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                
            img_t = to_numpy_img(recon_t_tensor)
            img_next_gt = to_numpy_img(recon_next_tensor)
            img_hat = to_numpy_img(recon_hat_tensor)
            img_orig_curr = to_numpy_img(img_curr)
            img_orig_next = to_numpy_img(img_next)
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            axes[0, 0].imshow(img_orig_curr)
            axes[0, 0].set_title("Input $O_t$")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(img_orig_next)
            axes[0, 1].set_title(f"Input $O_{{t+1}}$\n(Action: {sample['action']})")
            axes[0, 1].axis('off')
            
            axes[0, 2].axis('off') # Blank
            
            axes[1, 0].imshow(img_t)
            axes[1, 0].set_title("Recon $O_t$\n(from $Z_t$)")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(img_next_gt)
            axes[1, 1].set_title("Recon $O_{{t+1}}$ GT\n(from $Z_{{t+1}}$)")
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(img_hat)
            axes[1, 2].set_title(r"Hallucination $\hat{O}_{t+1}$" + "\n(from Transition Model)")
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(data_root, f"hallucinated_forward_{i+1}.jpg")
            plt.savefig(save_path, dpi=150)
            plt.close(fig) # close it so they don't overlap in memory
            print(f"Saved visualization to: {save_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Visualization failed: {e}")

if __name__ == '__main__':
    evaluate_forward_prediction()
