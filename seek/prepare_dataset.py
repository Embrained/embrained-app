import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cognitive-engine"))

from backend.training import TrainingPipeline
from backend.services.datasets import DatasetService
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PrepareDataset")

def main():
    logger.info("Initializing Dataset Preparation...")
    
    # Instantiate services
    dataset_service = DatasetService(DATA_DIR)
    pipeline = TrainingPipeline(DATA_DIR)
    
    # 1. Fetch all available datasets
    logger.info(f"Scanning {DATA_DIR} for datasets...")
    datasets_info = dataset_service.list_datasets(fast=True)
    dataset_names = [d['name'] for d in datasets_info.get('datasets', [])]
    
    if not dataset_names:
        logger.error("No datasets found in data directory. Please record some data first.")
        return
        
    logger.info(f"Found {len(dataset_names)} valid recording datasets. Pooling transitions...")
    
    # 2. Trigger the processing pipeline
    # We pass the explicit list of names so it pools them all together
    result = pipeline.process_datasets(dataset_names, extract_goals=True, extract_telemetry=True)
    
    if result.get("status") == "success":
        logger.info("\n" + "="*50)
        logger.info("✅ DATASET PREPARATION COMPLETE ✅")
        logger.info("="*50)
        logger.info("The transitions have been pooled and saved to: all_transitions.json")
        logger.info("You can now run any of the training scripts!")
    else:
        logger.error(f"Dataset preparation failed: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
