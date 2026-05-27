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

import sys
import logging
sys.path.append(r"c:\Users\chris\Embrained\software_suite")
from backend.training import TrainingPipeline

logging.basicConfig(level=logging.DEBUG)

def test_cql_gt():
    # Pass parent data directory so _expand_datasets works
    pipeline = TrainingPipeline(r"c:\Users\chris\Embrained\software_suite\data")
    
    class DummyEvent:
        def is_set(self): return False
        
    def progress(*args, **kwargs):
        pass
        
    try:
        res = pipeline.run_cql_pipeline(
            num_epochs=1, 
            stop_event=DummyEvent(),
            progress_callback=progress,
            vae_model="master_telemetry.csv", 
            batch_size=64,
            learning_rate=0.0001,
            model_size="small",
            dataset_percent=10,
            goal_type="her",
            selected_datasets=["markov_2026-03-22_14-47-27"],
            model_filename="test_cql_gt.pth"
        )
        print("Success:", res)
    except Exception as e:
        print("Crash:", str(e))

if __name__ == "__main__":
    test_cql_gt()
