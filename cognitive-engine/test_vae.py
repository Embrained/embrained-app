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

import logging
logging.basicConfig(level=logging.INFO)
from backend.training import TrainingPipeline

print("[TEST] Starting pipeline...")
pipeline = TrainingPipeline("C:/Users/chris/Embrained/software_suite/data")
print("[TEST] Running VAE pipeline...")
pipeline.run_vae_pipeline(num_epochs=1, batch_size=64)
print("[TEST] Done.")
