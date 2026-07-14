import os
import sys
import glob
import pandas as pd
import numpy as np
import ast
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), 'cognitive-engine'))
sys.path.append(os.path.join(os.getcwd(), 'seek'))
from modules.spatial_model import ContrastiveVisuomotorEncoder

DATA_ROOT = 'data'
SESSION_DIR = 'data/markov_2026-07-08_13-15-56'
GOAL_DIR = 'data/sofa'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading CVE...")
cve_candidates = glob.glob(os.path.join(DATA_ROOT, 'cve_*.pth'))
cve_candidates = [f for f in cve_candidates if not any(x in f.lower() for x in ['cql', 'policy', 'classifier'])]
cve_candidates.sort(key=os.path.getmtime, reverse=True)
cve_path = cve_candidates[0]

state_dict = torch.load(cve_path, map_location=device, weights_only=True)
encoder = ContrastiveVisuomotorEncoder(latent_dim=32, model_size='large').to(device)
encoder.load_state_dict(state_dict)
encoder.eval()

transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

print("Computing Goal Centroid...")
goal_imgs = glob.glob(os.path.join(GOAL_DIR, '*.jpg'))
goal_tensors = []
for f in goal_imgs:
    try:
        img = Image.open(f).convert('RGB')
        goal_tensors.append(transform(img))
    except: pass
goal_batch = torch.stack(goal_tensors).to(device)
with torch.no_grad():
    goal_latents = encoder.encode(goal_batch).cpu().numpy()
centroid = goal_latents.mean(axis=0)

print("Processing Session Data...")
csv_path = os.path.join(SESSION_DIR, 'episode_data.csv')
df = pd.read_csv(csv_path)

def get_action(row):
    act_raw = str(row.get('action_id', '0'))
    try: return int(act_raw)
    except:
        try:
            tup = ast.literal_eval(act_raw)
            if tup == (0, 0): return 5
            return 1
        except: return 1

df['macro_action'] = df.apply(get_action, axis=1)

actions = df['macro_action'].tolist()
cleaned_actions = []
n = len(actions)
for i in range(n):
    if actions[i] == 5:
        is_valid_stop = False
        if i > 0 and actions[i-1] == 5: is_valid_stop = True
        if i < n - 1 and actions[i+1] == 5: is_valid_stop = True
        if is_valid_stop: cleaned_actions.append(5)
        else: cleaned_actions.append(1)
    else: cleaned_actions.append(1)
df['cleaned_action'] = cleaned_actions

bouts = []
current_bout_frames = []

for idx, row in df.iterrows():
    action = row['cleaned_action']
    img_file = row.get('image_file', '')
    img_path = os.path.join(SESSION_DIR, 'images', img_file)
    
    if action != 5:
        if os.path.exists(img_path):
            current_bout_frames.append(img_path)
    else:
        if len(current_bout_frames) > 0:
            bouts.append(current_bout_frames)
            current_bout_frames = []

print(f"Found {len(bouts)} approach bouts.")

last_20_dists_list = []

for i, bout in enumerate(bouts):
    if len(bout) < 20: continue
    
    # Take only the last 20 frames
    last_20 = bout[-20:]
    
    tensors = []
    for f in last_20:
        img = Image.open(f).convert('RGB')
        tensors.append(transform(img))
    
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        latents = encoder.encode(batch).cpu().numpy()
    
    dists = np.linalg.norm(latents - centroid, axis=1)
    last_20_dists_list.append(dists)

print(f"Found {len(last_20_dists_list)} bouts that are >= 20 steps long.")

if len(last_20_dists_list) > 0:
    all_dists_arr = np.array(last_20_dists_list)  # Shape: (N, 20)
    avg_dists = all_dists_arr.mean(axis=0)
    std_dists = all_dists_arr.std(axis=0)
    
    plt.figure(figsize=(10, 6))
    
    x_steps = np.arange(-20, 0)
    
    # Plot individual lines lightly in background
    for dists in all_dists_arr:
        plt.plot(x_steps, dists, alpha=0.15, color='gray')
        
    # Plot the average
    plt.plot(x_steps, avg_dists, color='blue', linewidth=3, label='Average')
    plt.fill_between(x_steps, avg_dists - std_dists, avg_dists + std_dists, color='blue', alpha=0.2, label='Standard Deviation')
    
    plt.title('Average Latent Distance to Goal Centroid (Last 20 Steps)')
    plt.xlabel('Steps until arrival (Intentional Stop)')
    plt.ylabel('L2 Latent Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.xticks(np.arange(-20, 1, 2))
    
    plt.tight_layout()
    out_plot = 'C:\\Users\\chris\\Embrained\\images\\approach_latent_distances_avg20.png'
    plt.savefig(out_plot, dpi=150)
    print(f"Saved average plot to {out_plot}")
else:
    print("No bouts were 20 steps or longer.")
