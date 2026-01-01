
import os
import sys
import glob
import logging
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm

# Add parent directory to sys.path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules.vision import VisionSystem
except ImportError:
    # If running from scripts/, parent is one level up
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.vision import VisionSystem

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_directory(vision_system, capture_dir, specific_files=None):
    # specific_files: Optional list of filenames (basenames) to process in this directory.
    # If None, process all.
    
    logging.info(f"Processing directory: {capture_dir}")
    
    if specific_files is not None:
        # Construct full paths from basenames
        jpg_files = [os.path.join(capture_dir, f) for f in specific_files]
        # Verify existence just in case
        jpg_files = [f for f in jpg_files if os.path.exists(f)]
        if not jpg_files:
            logging.warning(f"No matching files found in {capture_dir} from the sample list.")
            return
        desc_text = f"Processing {len(jpg_files)} sampled files in {os.path.basename(capture_dir)}"
    else:
        # Find all .jpg files
        jpg_files = sorted(glob.glob(os.path.join(capture_dir, "*.jpg")))
        if not jpg_files:
            logging.warning(f"No .jpg files found in {capture_dir}")
            return
        desc_text = f"Processing all files in {os.path.basename(capture_dir)}"
        
    latents = []
    frame_names = []
    
    for img_path in tqdm(jpg_files, desc=desc_text):
        try:
            # Read image using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue
                
            # Process frame using VisionSystem
            _, latent = vision_system.process_frame(img)
            
            if latent is not None:
                latents.append(latent)
                frame_names.append(os.path.basename(img_path))
            else:
                logging.warning(f"Failed to generate latent for: {img_path}")
                
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")

    if latents:
        # Convert to numpy array
        latents_array = np.array(latents, dtype=np.float32)
        
        # Save latents.npy
        output_path = os.path.join(capture_dir, "latents.npy")
        np.save(output_path, latents_array)
        logging.info(f"Saved {len(latents)} latents to {output_path}")
        
        # Save frames.json
        import json
        frames_path = os.path.join(capture_dir, "frames.json")
        with open(frames_path, 'w') as f:
            json.dump(frame_names, f, indent=2)
    else:
        logging.warning("No latents generated.")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Generate MobileNetV3 latents for capture directories.")
    parser.add_argument("--root_dir", type=str, default=r"C:\Users\chris\ArtificialBrain\Explorer", help="Root directory containing capture folders")
    parser.add_argument("-n", "--num_samples", type=int, default=None, help="Number of random samples to process across all directories. If None, process all.")
    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        logging.error(f"Root directory does not exist: {args.root_dir}")
        return

    # Find capture directories
    capture_dirs = [d for d in glob.glob(os.path.join(args.root_dir, "capture-*")) if os.path.isdir(d)]
    
    if not capture_dirs:
        logging.warning(f"No 'capture-*' directories found in {args.root_dir}")
        return
        
    logging.info(f"Found {len(capture_dirs)} capture directories.")

    # Sampling Logic
    files_to_process_by_dir = {} # Dict[dir_path] -> List[filename] (or None for all)
    
    if args.num_samples is not None:
        logging.info(f"Gathering file lists from all directories to sample {args.num_samples} frames...")
        all_files_map = [] # List of (dir_path, filename)
        
        for d in capture_dirs:
            # We assume simple filenames, but let's grab them
            # using glob in each dir
            fnames = [os.path.basename(p) for p in glob.glob(os.path.join(d, "*.jpg"))]
            for fn in fnames:
                all_files_map.append((d, fn))
                
        total_files = len(all_files_map)
        logging.info(f"Total available files: {total_files}")
        
        if total_files == 0:
            logging.warning("No files found to sample.")
            return
            
        if args.num_samples >= total_files:
            logging.info(f"Requested samples {args.num_samples} >= total files {total_files}. Processing ALL.")
            # files_to_process_by_dir remains empty/None logic -> process all
            pass
        else:
            # Random sample indices
            import random
            # Shuffle indices
            indices = list(range(total_files))
            random.shuffle(indices)
            selected_indices = indices[:args.num_samples]
            
            # Reorganize selection
            for idx in selected_indices:
                d, fn = all_files_map[idx]
                if d not in files_to_process_by_dir:
                    files_to_process_by_dir[d] = []
                files_to_process_by_dir[d].append(fn)
                
            logging.info(f"Selected {len(selected_indices)} files across {len(files_to_process_by_dir)} directories.")

    # Initialize VisionSystem
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Initializing VisionSystem on {device}...")
    try:
        vision = VisionSystem(device=device)
    except Exception as e:
        logging.error(f"Failed to initialize VisionSystem: {e}")
        return
    
    for capture_dir in capture_dirs:
        # If we have a sampling plan
        if args.num_samples is not None:
            # If this dir is not in our map, it has 0 samples, skip it
            if files_to_process_by_dir and capture_dir not in files_to_process_by_dir:
                pass # Skip
            elif not files_to_process_by_dir and args.num_samples < total_files:
                 # Case where we sampled but this dir got nothing?
                 # Handled by 'capture_dir not in files_to_process_by_dir'
                 pass
            else:
                # We have specific files, or we are processing all
                # If files_to_process_by_dir is empty but we fell through, it implies process ALL (num_samples >= total)
                specific = files_to_process_by_dir.get(capture_dir, None)
                process_directory(vision, capture_dir, specific_files=specific)
        else:
            # Process all
            process_directory(vision, capture_dir)

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
