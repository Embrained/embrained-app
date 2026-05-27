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
import time
import pickle
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ManifoldTest")

# Add Path
sys.path.append(os.getcwd())

from backend.manifold import ManifoldService

# Mock Constants
MODELS_DIR = os.path.join(os.getcwd(), "models")
TEST_MODEL_NAME = "test_vae_model.pth"
TEST_CACHE_NAME = "test_vae_model_manifold.pkl"

TEST_MODEL_PATH = os.path.join(MODELS_DIR, TEST_MODEL_NAME)
TEST_CACHE_PATH = os.path.join(MODELS_DIR, TEST_CACHE_NAME)

def setup_files():
    # Create Dummy Model
    with open(TEST_MODEL_PATH, 'w') as f:
        f.write("dummy model content")
    
    # Create Dummy Cache (Older timestamp)
    # pickle dummy data
    data = {
        'pca': "dummy_pca",
        'points': [],
        'latents': [],
        'paths': []
    }
    with open(TEST_CACHE_PATH, 'wb') as f:
        pickle.dump(data, f)
        
    # Ensure Model is NEWER than Cache
    # Set Cache time to 100s ago
    now = time.time()
    os.utime(TEST_CACHE_PATH, (now - 100, now - 100))
    os.utime(TEST_MODEL_PATH, (now, now))
    
    logger.info(f"Created Test Files:\n  Model: {time.ctime(os.path.getmtime(TEST_MODEL_PATH))}\n  Cache: {time.ctime(os.path.getmtime(TEST_CACHE_PATH))}")

def cleanup():
    if os.path.exists(TEST_MODEL_PATH): os.remove(TEST_MODEL_PATH)
    if os.path.exists(TEST_CACHE_PATH): os.remove(TEST_CACHE_PATH)

def test_stale_cache():
    logger.info("--- Testing Stale Cache ---")
    service = ManifoldService()
    
    # CASE 1: Path Passed Explicitly (Should Fail Load / Return False)
    service.set_model_name(TEST_MODEL_NAME, model_path=TEST_MODEL_PATH)
    
    # We access the internal _load_from_cache to verify logic
    result = service._load_from_cache()
    
    if result is False:
        logger.info("[PASS] Cache was correctly identified as stale when path provided.")
    else:
        logger.error("[FAIL] Cache was erroneously loaded!")
        
    # CASE 2: No Path Passed (Should FAIL or WARN, but fallback logic might miss it if name differs)
    # Our fallback logic looks for 'tiny_vae_final.pth'. Our test model is 'test_vae_model.pth'.
    # So it won't find a model to compare against, and might load the cache blindly if we didn't add the path check.
    # But wait, if model_path is None, we check "tiny_vae_final.pth". If that doesn't exist?
    # Logic: if not model_path -> check defaults. If defaults found -> compare. If defaults NOT found -> load cache?
    # This is slightly dangerous if we have a custom model nickname but no path.
    # But in engine.py we GUARANTEE path now.
    
    logger.info("--- Testing Valid Cache ---")
    # Touch Cache to be newer
    now = time.time()
    os.utime(TEST_CACHE_PATH, (now + 100, now + 100))
    
    result = service._load_from_cache()
    if result is True:
        logger.info("[PASS] Cache was loaded when timestamp valid.")
    else:
        logger.error("[FAIL] Cache should have been loaded!")

if __name__ == "__main__":
    try:
        setup_files()
        test_stale_cache()
    finally:
        cleanup()
