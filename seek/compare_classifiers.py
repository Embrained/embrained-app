import os
import sys
import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'cognitive-engine'))
sys.path.append(os.path.join(os.getcwd(), 'seek'))
from modules.goal_classifier import GoalClassifier

def main():
    DATA_ROOT = 'data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sofa_path = os.path.join(DATA_ROOT, 'sofa_classifier.pth')
    bookshelf_path = os.path.join(DATA_ROOT, 'bookshelf_classifier.pth')

    print(f"Loading classifiers on {device}...")
    sofa_classifier = GoalClassifier.load_from_checkpoint(sofa_path, device=device)
    sofa_classifier.eval()
    
    bookshelf_classifier = GoalClassifier.load_from_checkpoint(bookshelf_path, device=device)
    bookshelf_classifier.eval()

    transitions_file = os.path.join(DATA_ROOT, 'all_transitions.json')
    print(f"Loading dataset {transitions_file}...")
    with open(transitions_file, 'r') as f:
        transitions = json.load(f)
    
    print(f"Total transitions: {len(transitions)}")
    
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    sofa_probs = []
    bookshelf_probs = []

    batch_size = 128
    batch_tensors = []

    def process_batch(tensors):
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            s_probs = sofa_classifier.predict_proba(batch).cpu().numpy()
            b_probs = bookshelf_classifier.predict_proba(batch).cpu().numpy()
        return s_probs, b_probs

    print("Running images through classifiers...")
    for t in tqdm(transitions):
        img_rel_path = t.get('image_path')
        if not img_rel_path:
            continue
            
        img_path = os.path.join(DATA_ROOT, img_rel_path)
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            batch_tensors.append(tensor)
        except Exception as e:
            continue
            
        if len(batch_tensors) >= batch_size:
            s_p, b_p = process_batch(batch_tensors)
            sofa_probs.extend(s_p)
            bookshelf_probs.extend(b_p)
            batch_tensors = []

    if len(batch_tensors) > 0:
        s_p, b_p = process_batch(batch_tensors)
        sofa_probs.extend(s_p)
        bookshelf_probs.extend(b_p)

    sofa_probs = np.array(sofa_probs)
    bookshelf_probs = np.array(bookshelf_probs)
    
    print(f"Total processed: {len(sofa_probs)}")
    
    # Calculate how many frames exceed training threshold 0.90
    sofa_terminals = np.sum(sofa_probs > 0.90)
    bookshelf_terminals = np.sum(bookshelf_probs > 0.90)
    
    print(f"Sofa frames > 0.90: {sofa_terminals}")
    print(f"Bookshelf frames > 0.90: {bookshelf_terminals}")

    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(sofa_probs, bins=50, color='blue', alpha=0.7)
    ax1.axvline(x=0.90, color='red', linestyle='--', label='Threshold (0.90)')
    ax1.set_title('Sofa Classifier Probabilities')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log') # Log scale helps see the small counts at the high end
    ax1.legend()

    ax2.hist(bookshelf_probs, bins=50, color='green', alpha=0.7)
    ax2.axvline(x=0.90, color='red', linestyle='--', label='Threshold (0.90)')
    ax2.set_title('Bookshelf Classifier Probabilities')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Count')
    ax2.set_yscale('log')
    ax2.legend()

    plt.suptitle('Goal Classifier Output Distributions Across Entire Exploration Dataset (Log Scale)')
    plt.tight_layout()
    
    out_img = 'C:\\Users\\chris\\Embrained\\images\\classifier_comparison_hist.png'
    plt.savefig(out_img, dpi=150)
    print(f"Plot saved to {out_img}")

if __name__ == '__main__':
    main()
