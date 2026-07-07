# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# Evaluation script for navigation trials.
# Automatically scores inference sessions from manual displacement trials.
#
# Usage:
#   python evaluate_navigation_trials.py <session_dir> [--classifier sofa_classifier.pth]
#
# A trial session consists of multiple manual displacements:
#   1. User places robot at a random location
#   2. Policy runs until INTENTIONAL_STOP or max_steps
#   3. User displaces robot again (detected by embedding discontinuity)

import os
import sys
import json
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cognitive-engine')))
from config import DATA_DIR
from modules.spatial_model import TinyVAE, ContrastiveVisuomotorEncoder
from modules.goal_classifier import GoalClassifier


def load_cve_encoder(data_root, device):
    """Load the latest CVE encoder from the data directory."""
    cve_candidates = glob.glob(os.path.join(data_root, 'cve_*.pth'))
    cve_candidates = [f for f in cve_candidates if not any(x in f.lower() for x in ['cql', 'policy', 'classifier'])]
    if not cve_candidates:
        raise FileNotFoundError("No CVE model found in data directory")
    
    cve_candidates.sort(key=os.path.getmtime, reverse=True)
    cve_path = cve_candidates[0]
    
    state_dict = torch.load(cve_path, map_location=device, weights_only=True)
    latent_dim, model_size, img_dim, in_channels = TinyVAE.detect_size(state_dict)
    n_actions = state_dict['action_predictor.2.weight'].shape[0]
    
    encoder = ContrastiveVisuomotorEncoder(
        latent_dim=latent_dim, model_size=model_size,
        input_spatial_dim=img_dim, in_channels=in_channels,
        n_actions=n_actions
    ).to(device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    print(f"Loaded CVE encoder: {os.path.basename(cve_path)} (latent={latent_dim}d, img={img_dim})")
    return encoder, img_dim


def segment_trials(session_data, embeddings, max_steps=50, 
                   displacement_threshold=1.0):
    """Segment a continuous recording session into individual navigation trials.
    
    Trial boundaries are detected by:
    1. INTENTIONAL_STOP actions (macro_action=5) → end of current trial
    2. Embedding discontinuity → manual displacement → new trial start
    3. Timeout after max_steps → failed trial
    
    Args:
        session_data: List of transition dicts (sorted by timestamp)
        embeddings: numpy array [N, latent_dim] of CVE embeddings
        max_steps: Maximum steps before timeout
        displacement_threshold: CVE distance threshold for manual displacement detection
    
    Returns:
        List of trial dicts with start_idx, end_idx, steps, end_reason
    """
    trials = []
    current_trial_start = 0
    n = len(session_data)
    
    # State machine for trial segmentation
    # state can be 'in_trial' or 'waiting_for_displacement'
    state = 'in_trial'
    
    i = 0
    while i < n:
        # Check for displacement
        if i > 0:
            dist = np.linalg.norm(embeddings[i] - embeddings[i-1])
            if dist > 1.0:
                print(f"Frame {i}: Distance from previous = {dist:.2f}")
            if dist > displacement_threshold:
                # Manual displacement detected — start a new trial
                if state == 'in_trial' and current_trial_start < i:
                    trials.append({
                        'start_idx': current_trial_start,
                        'end_idx': i - 1,
                        'steps': i - current_trial_start,
                        'end_reason': 'timeout',  # Previous trial didn't stop
                    })
                current_trial_start = i
                state = 'in_trial'
        
        if state == 'in_trial':
            # Check for INTENTIONAL_STOP
            action = session_data[i].get('macro_action', 0)
            if action == 5:
                trials.append({
                    'start_idx': current_trial_start,
                    'end_idx': i,
                    'steps': i - current_trial_start + 1,
                    'end_reason': 'stop',
                })
                state = 'waiting_for_displacement'
            
            # Check for timeout
            elif (i - current_trial_start) >= max_steps:
                trials.append({
                    'start_idx': current_trial_start,
                    'end_idx': i,
                    'steps': i - current_trial_start + 1,
                    'end_reason': 'timeout',
                })
                state = 'waiting_for_displacement'
        
        i += 1
    
    # Handle any remaining frames as an incomplete trial
    if current_trial_start < n:
        trials.append({
            'start_idx': current_trial_start,
            'end_idx': n - 1,
            'steps': n - current_trial_start,
            'end_reason': 'incomplete',
        })
    
    return trials


def evaluate_session(session_dir, classifier, encoder, img_dim, device,
                     classifier_threshold=0.85, max_steps=50):
    """Evaluate all navigation trials in a recorded session.
    
    Args:
        session_dir: Path to the markov_* session directory
        classifier: Trained GoalClassifier model
        encoder: CVE encoder for embedding computation
        img_dim: Image dimension for resizing
        device: torch device
        classifier_threshold: P(goal) threshold for success
        max_steps: Max steps per trial
    
    Returns:
        trial_results: List of per-trial result dicts
        summary: Aggregate statistics dict
    """
    # Load session data
    csv_path = os.path.join(session_dir, 'episode_data.csv')
    if not os.path.exists(csv_path):
        print(f"No episode_data.csv found in {session_dir}")
        return [], {}
    
    import csv
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if not rows:
        return [], {}
    
    print(f"Session: {os.path.basename(session_dir)} ({len(rows)} frames)")
    
    # Load and encode all images
    transform = T.Compose([T.Resize((img_dim, img_dim)), T.ToTensor()])
    img_dir = os.path.join(session_dir, 'images')
    
    embeddings = []
    classifier_scores = []
    valid_rows = []
    
    batch_imgs = []
    batch_indices = []
    
    for idx, row in enumerate(rows):
        img_file = row.get('image_file', '')
        img_path = os.path.join(img_dir, img_file)
        
        if not os.path.exists(img_path):
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            batch_imgs.append(tensor)
            batch_indices.append(idx)
            valid_rows.append(row)
        except Exception:
            continue
        
        # Process in batches
        if len(batch_imgs) >= 64:
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                z = encoder.encode(batch).cpu().numpy()
                scores = classifier.predict_proba(batch).cpu().numpy()
            embeddings.append(z)
            classifier_scores.append(scores)
            batch_imgs = []
    
    # Process remaining
    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            z = encoder.encode(batch).cpu().numpy()
            scores = classifier.predict_proba(batch).cpu().numpy()
        embeddings.append(z)
        classifier_scores.append(scores)
    
    if not embeddings:
        return [], {}
    
    all_embeddings = np.concatenate(embeddings, axis=0)
    all_scores = np.concatenate(classifier_scores, axis=0)
    
    print(f"  Encoded {len(all_embeddings)} frames")
    print(f"  Classifier scores: min={all_scores.min():.3f}, max={all_scores.max():.3f}, mean={all_scores.mean():.3f}")
    
    # Segment into trials
    # Add macro_action to valid_rows from CSV data
    for row in valid_rows:
        pwm_l = float(row.get('pwm_left', 0))
        pwm_r = float(row.get('pwm_right', 0))
        # Detect STOP (both motors ~0)
        if abs(pwm_l) < 10 and abs(pwm_r) < 10:
            row['macro_action'] = 5  # INTENTIONAL_STOP
        else:
            row['macro_action'] = 1  # Some movement action
    
    trials = segment_trials(valid_rows, all_embeddings, max_steps=max_steps)
    
    print(f"  Detected {len(trials)} trials")
    
    # Score each trial
    trial_results = []
    for t_idx, trial in enumerate(trials):
        start = trial['start_idx']
        end = trial['end_idx']
        
        # Get classifier score at the final frame
        final_score = float(all_scores[end])
        
        # Check if the trial was successful
        success = final_score >= classifier_threshold and trial['end_reason'] == 'stop'
        
        # Also check max score during the trial (robot may have reached goal but kept going)
        trial_scores = all_scores[start:end+1]
        max_score = float(np.max(trial_scores))
        
        result = {
            'trial_id': t_idx + 1,
            'steps': trial['steps'],
            'success': success,
            'end_reason': trial['end_reason'],
            'final_confidence': round(final_score, 4),
            'max_confidence': round(max_score, 4),
            'start_frame': valid_rows[start].get('image_file', ''),
            'end_frame': valid_rows[end].get('image_file', ''),
        }
        trial_results.append(result)
        
        status = "OK" if success else "FAIL"
        print(f"  Trial {t_idx+1}: {status} steps={trial['steps']:2d} | "
              f"final_P={final_score:.3f} | max_P={max_score:.3f} | "
              f"reason={trial['end_reason']}")
    
    # Compute summary
    num_trials = len(trial_results)
    num_success = sum(1 for r in trial_results if r['success'])
    success_rate = num_success / max(num_trials, 1)
    
    successful_steps = [r['steps'] for r in trial_results if r['success']]
    
    summary = {
        'session': os.path.basename(session_dir),
        'num_frames': len(all_embeddings),
        'num_trials': num_trials,
        'num_success': num_success,
        'success_rate': round(success_rate, 4),
        'mean_steps_successful': round(np.mean(successful_steps), 1) if successful_steps else 0,
        'median_steps_successful': round(float(np.median(successful_steps)), 1) if successful_steps else 0,
        'mean_confidence_all': round(float(all_scores.mean()), 4),
    }
    
    return trial_results, summary


def plot_trial_results(trial_results, summary, output_path):
    """Generate a visual summary of trial results."""
    if not trial_results:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Navigation Trial Results — {summary["session"]}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Steps per trial (bar chart)
    trial_ids = [r['trial_id'] for r in trial_results]
    steps = [r['steps'] for r in trial_results]
    colors = ['#4ECDC4' if r['success'] else '#FF6B6B' for r in trial_results]
    
    axes[0].bar(trial_ids, steps, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Steps')
    axes[0].set_title('Steps per Trial')
    axes[0].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Target (20)')
    axes[0].legend()
    
    # 2. Classifier confidence at end (bar chart)
    final_confs = [r['final_confidence'] for r in trial_results]
    axes[1].bar(trial_ids, final_confs, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='Threshold (0.85)')
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('P(goal)')
    axes[1].set_title('Final Classifier Confidence')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    
    # 3. Summary text
    axes[2].axis('off')
    summary_text = (
        f"Trials: {summary['num_trials']}\n"
        f"Successes: {summary['num_success']}\n"
        f"Success Rate: {summary['success_rate']*100:.1f}%\n"
        f"\n"
        f"Mean Steps (success): {summary['mean_steps_successful']}\n"
        f"Median Steps (success): {summary['median_steps_successful']}\n"
        f"\n"
        f"Mean Confidence: {summary['mean_confidence_all']:.3f}"
    )
    axes[2].text(0.1, 0.5, summary_text, transform=axes[2].transAxes,
                fontsize=13, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].set_title('Summary')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved trial results plot to {output_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT = os.path.abspath(DATA_DIR)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python evaluate_navigation_trials.py <session_dir_or_name> [--classifier <path>] [--goal <subdir>]")
        print("\nExample:")
        print("  python evaluate_navigation_trials.py markov_2026-07-03_14-30-00")
        print("  python evaluate_navigation_trials.py markov_2026-07-03_14-30-00 --goal sofa")
        return
    
    session_input = sys.argv[1]
    goal_subdir = 'sofa'
    classifier_path = None
    
    # Parse optional args
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--classifier' and i + 1 < len(sys.argv):
            classifier_path = sys.argv[i + 1]
        elif arg == '--goal' and i + 1 < len(sys.argv):
            goal_subdir = sys.argv[i + 1]
    
    # Resolve session directory
    if os.path.isabs(session_input):
        session_dir = session_input
    elif os.path.isdir(os.path.join(DATA_ROOT, session_input)):
        session_dir = os.path.join(DATA_ROOT, session_input)
    else:
        # Try glob match
        matches = glob.glob(os.path.join(DATA_ROOT, f'*{session_input}*'))
        if matches:
            session_dir = matches[0]
        else:
            print(f"ERROR: Could not find session: {session_input}")
            return
    
    print(f"Session directory: {session_dir}")
    
    # Load models
    if classifier_path is None:
        classifier_path = os.path.join(DATA_ROOT, f'{goal_subdir}_classifier.pth')
    
    if not os.path.exists(classifier_path):
        print(f"ERROR: Classifier not found at {classifier_path}")
        return
    
    classifier = GoalClassifier.load_from_checkpoint(classifier_path, device=str(device))
    encoder, img_dim = load_cve_encoder(DATA_ROOT, device)
    
    # Evaluate
    trial_results, summary = evaluate_session(
        session_dir, classifier, encoder, img_dim, device,
        classifier_threshold=0.85, max_steps=50
    )
    
    if not trial_results:
        print("No trials detected in this session.")
        return
    
    # Save results
    results_path = os.path.join(DATA_ROOT, 'trial_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': summary,
            'trials': trial_results,
        }, f, indent=2)
    print(f"\n[OK] Saved results JSON to {results_path}")
    
    # Plot
    images_dir = os.path.join(os.path.dirname(DATA_ROOT), 'images')
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, 'trial_results.png')
    plot_trial_results(trial_results, summary, plot_path)
    
    # Final summary
    print(f"\n{'='*50}")
    print(f" EVALUATION COMPLETE")
    print(f"{'='*50}")
    print(f" Trials: {summary['num_trials']}")
    print(f" Success Rate: {summary['success_rate']*100:.1f}%")
    if summary['mean_steps_successful'] > 0:
        print(f" Mean Steps (success): {summary['mean_steps_successful']}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
