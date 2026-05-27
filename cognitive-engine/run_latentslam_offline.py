import os
import sys
import logging
import time
from pathlib import Path

# Fix relative imports when running from root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.training import TrainingPipeline
from config import DATA_DIR, MODELS_DIR

# --- 1. Verbose Logging Configuration ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(DATA_DIR, "standalone_latentslam.log"), mode='w')
    ]
)
logger = logging.getLogger("StandaloneTrainer")

# Silence noisy matplotlib/PIL debugs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def progress_spy(epoch, loss, kl=0.0, recon=0.0, manifold_plot=None, **kwargs):
    """Verbose local progress interceptor."""
    logger.info(f"[EPOCH {epoch:03d}] Loss: {loss:.4f} | KL: {kl:.4f} | Recon: {recon:.4f}")
    if manifold_plot:
        logger.info(f" -> Manifold Dashboard successfully materialized!")

def main():
    logger.info("==================================================")
    logger.info("   ROBUST LATENT-SLAM OFFLINE TRAINING SEQUENCE   ")
    logger.info("==================================================")
    
    # --- 2. Automated Dataset Scavenging ---
    data_root = Path(DATA_DIR)
    if not data_root.exists():
        logger.error(f"Data root {DATA_DIR} does not exist!")
        return
        
    all_dirs = [d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Filter out non-dataset utility folders
    ignore_list = ["goals", "__pycache__", "recordings"]
    dataset_dirs = [d for d in all_dirs if d not in ignore_list and "_goals" not in d]
    
    logger.info(f"Discovered {len(dataset_dirs)} raw dataset candidate(s): {dataset_dirs}")
    if not dataset_dirs:
        logger.error("No valid telemetry datasets found to train on. Aborting.")
        return

    # --- 3. Pipeline Bootstrapping ---
    pipeline = TrainingPipeline(data_root=DATA_DIR)
    
    logger.info("\n>>> PHASE 1: Forcing Rigorous Data Pre-Processing & Transition Extraction")
    # By forcing extraction, we guarantee all telemetry boundaries are fresh and un-corrupted by UI interruptions
    prep_status = pipeline.process_datasets(dataset_dirs, extract_goals=True, extract_telemetry=True)
    if prep_status.get("status") != "success":
        logger.error(f"Data processing failed: {prep_status}")
        return
    logger.info(f"Data processing complete: {prep_status.get('message', 'OK')}")

    # --- 4. Secure Hyperparameter Configuration ---
    # We configure this aggressively for maximal stable convergence based on previous successful traces
    config = {
        "num_epochs": 40,
        "batch_size": 128,          # Scaled up for stable gradient steps
        "learning_rate": 1e-4,
        "beta": 2.0,                # High beta enforces strong continuous latent disentanglement
        "transition_loss_weight": 1.0,
        "contrastive_weight": 0.05, # Mild contrastive repelling to separate clusters
        "architecture": "continuous",
        "model_size": "large",      # 5x layered geometric depth
        "image_size": 64, 
        "latent_dim": 128,          # Maximal topological resolution
        "num_layers": 5,
        "selected_datasets": dataset_dirs
    }
    
    logger.info("\n>>> PHASE 2: Initializing LatentSLAM Graph Core")
    logger.info(f"Injecting Parameters: {config}")
    
    start_time = time.time()
    
    # --- 5. Fire Training sequence ---
    result = pipeline.run_latentslam_pipeline(
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        beta=config["beta"],
        transition_loss_weight=config["transition_loss_weight"],
        contrastive_weight=config["contrastive_weight"],
        architecture=config["architecture"],
        model_size=config["model_size"],
        selected_datasets=config["selected_datasets"],
        image_size=config["image_size"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        progress_callback=progress_spy
    )
    
    elapsed = time.time() - start_time
    logger.info("==================================================")
    if result.get("status") == "success":
        logger.info(f"TRAINING COMPLETE IN {elapsed:.2f} seconds!")
        logger.info(f"Model confidently secured at: {result.get('model_path')}")
    else:
        logger.error(f"PIPELINE CRASHED: {result.get('message')}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
