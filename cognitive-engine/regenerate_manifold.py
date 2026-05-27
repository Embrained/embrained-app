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
import logging
from backend.manifold import ManifoldService
from modules.vision import VisionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RegenerateManifold")

def main():
    logger.info("Starting Manifold Regeneration...")
    
    # 1. Init Vision
    try:
        vision = VisionSystem(device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        logger.info(f"Vision initialized on {vision.device}")
    except Exception as e:
        logger.error(f"Vision Init Failed: {e}")
        return

    # 2. Init Manifold
    try:
        manifold = ManifoldService(vision)
        logger.info("Manifold Service initialized.")
    except Exception as e:
        logger.error(f"Manifold Init Failed: {e}")
        return

    # 3. Force Fit
    logger.info("Forcing Manifold Fit (Regenerating Cache)...")
    manifold.fit(force=True)
    logger.info("Manifold Regeneration Complete!")

if __name__ == "__main__":
    main()
