import time
import random
import logging

class ExplorationSystem:
    def __init__(self):
        self.current_algo = None
        self.last_action_time = 0
        self.current_action = 3 # Default STOP
        self.action_duration = 0
        self.state_start_time = 0
        
        # Definition of Actions
        # 0: FWD, 1: LEFT, 2: RIGHT, 3: STOP, 4: BACK
        self.actions = [0, 1, 2, 3, 4]
        
    def set_algorithm(self, name):
        """Sets the active exploration algorithm."""
        if name != self.current_algo:
            logging.info(f"Exploration Algorithm switched to: {name}")
            self.current_algo = name
            self._reset_state()
            
    def _reset_state(self):
        self.current_action = 3
        self.state_start_time = time.time()
        self.action_duration = 0

    def get_action(self):
        """Returns the next action based on the active algorithm."""
        if not self.current_algo:
            return 3 # STOP

        return self._process_explorer()

    def _process_explorer(self):
        now = time.time()
        elapsed = now - self.state_start_time
        
        # If current action is done, pick new one
        if elapsed >= self.action_duration:
            self._pick_new_action()
            
        return self.current_action

    def _pick_new_action(self):
        self.state_start_time = time.time()
        
        # Logic: Alternating Move -> Stop -> Move -> Stop
        
        # Check if we should Switch to STOP
        # If we are currently MOVING (not 3), our next state must be STOP.
        if self.current_action != 3:
            self.current_action = 3
            
            # Stop Duration depends on Algo
            if self.current_algo == "Explorer1":
                # Separated by 1-3s of Stop
                self.action_duration = random.uniform(1.0, 3.0)
            elif self.current_algo == "Explorer2":
                # Separated by 0.5-1.5s of Stop
                self.action_duration = random.uniform(0.5, 1.5)
            elif self.current_algo == "Rotator1":
                # Separated by 1-3s of Stop
                self.action_duration = random.uniform(1.0, 3.0)
            else:
                 # Default fallback
                self.action_duration = 1.0
                
            return

        # If we are currently STOPPED (3), we must Move.
        # Pick Move Action and Duration based on Algo
        
        if self.current_algo.startswith("Explorer"): 
            # Both Explorer 1 and 2 share movement probs and move duration
            # Forward 60%, Left 15%, Right 15%, Backward 10%
            # Actions: 0: FWD, 1: LEFT, 2: RIGHT, 4: BACK
            move_actions = [0, 1, 2, 4]
            probs = [0.60, 0.15, 0.15, 0.10]
            
            # Duration: 0.5 - 1s
            min_dur, max_dur = 0.5, 1.0
            
            new_action = random.choices(move_actions, weights=probs, k=1)[0]
            self.current_action = new_action
            self.action_duration = random.uniform(min_dur, max_dur)
            
        elif self.current_algo == "Rotator1":
            # Left 50%, Right 50%
            move_actions = [1, 2]
            probs = [0.5, 0.5]
            
            # Duration: 0.5 - 1s
            min_dur, max_dur = 0.5, 1.0
            
            new_action = random.choices(move_actions, weights=probs, k=1)[0]
            self.current_action = new_action
            self.action_duration = random.uniform(min_dur, max_dur)
            
        else:
            # Unknown algo, stay stopped
            self.current_action = 3
            self.action_duration = 1.0
