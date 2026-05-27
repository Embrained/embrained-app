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
import os
import time

# Add root to path
sys.path.append(os.getcwd())

import logging
logging.basicConfig(level=logging.INFO)

def test_engine_init():
    print("--- Testing Engine Initialization ---")
    try:
        from backend.engine import CognitiveEngine
        engine = CognitiveEngine(dry_run=True)
        print("Engine Instantiated.")
        
        # Check Components
        assert hasattr(engine, 'state_manager'), "Missing state_manager"
        assert hasattr(engine, 'model_manager'), "Missing model_manager"
        assert hasattr(engine, 'dispatcher'), "Missing dispatcher"
        print("Components Verified.")
        
        # Check Properties compatibility
        assert isinstance(engine.state, dict), "engine.state property failed"
        assert engine.state['mode'] == 'IDLE', "Default mode verify failed"
        
        # Test Locking
        print("Testing Lock Proxy...")
        with engine.state_lock:
             engine.state['fps'] = 30.0
        assert engine.state['fps'] == 30.0
        print("Lock Verified.")
        
        # Test Command Dispatch
        print("Testing Command Dispatch...")
        engine.handle_command("MOVE", 1) # Auto-Stop
        time.sleep(0.1) 
        # Since it's threaded queue, we wait a bit or check queue?
        # Engine loop processes queue. If we didn't start engine, queue sits there.
        # But we can check dispatcher directly.
        
        engine.dispatcher.dispatch("MOVE", 1)
        assert engine.current_live_action == 1, "Direct Dispatch failed"
        print("Command Dispatch Verified.")
        
        # Test Model Manager discovery (mock)
        print("Testing Model Discovery...")
        # Just check if it runs without error
        res = engine.model_manager.find_best_model("non_existent_model.pth")
        print(f"Model Search Result: {res} (Expected None)")

        print("\nSUCCESS: Refactor verification passed!")
        
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine_init()
