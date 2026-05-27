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

import time
import random
import logging
import numpy as np

# Action Definitions matched with config.py
ACTION_STOP = 0
ACTION_FWD = 1
ACTION_BACK = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

AUTONOMY_THRESHOLD = 2000

class MarkovSweep:
    """
    MarkovSweep: A 2D spatial momentum controller providing long-horizon sequences for HER.
    Uniformly selects a sweep duration of 1 to 5 steps, and holds an action (FWD, BACK, LEFT, RIGHT)
    for that duration before randomly switching direction and duration again.
    """
    def __init__(self):
        self.state = 'WAITING'
        self.state_start_time = 0
        self.current_action_id = 0
        self.sweep_target = 0
        self.sweep_count = 0
        self.is_backing = False
        self.action_queue = []

    def get_action(self, sensor_dist):
        try: dist = float(sensor_dist)
        except (ValueError, TypeError): dist = 0.0
        
        now = time.time()
        
        # Trigger Reflex
        if dist > AUTONOMY_THRESHOLD and not getattr(self, 'in_reflex', False):
            self.in_reflex = True
            self.action_queue = [] # clear any pending actions
            self.state = 'STOP'
            self.state_start_time = now
            
            turn_dir = random.choice([3, 4]) # LEFT or RIGHT action IDs
            self.action_queue.append(2) # single reversal first
            for _ in range(3):
                self.action_queue.append(turn_dir)

        if getattr(self, 'in_reflex', False) and len(self.action_queue) == 0:
            self.in_reflex = False
        
        if self.state == 'WAITING':
            if self.action_queue:
                self.current_action_id = self.action_queue.pop(0)
                self.sweep_target = 1  # 1 step of momentum for the queued reflex turn
                self.sweep_count = 0
            elif self.sweep_count >= self.sweep_target:
                # Distribution for picking a sweep sequence direction
                actions = [1, 2, 3, 4]
                probs = [0.4, 0.2, 0.2, 0.2] 
                self.current_action_id = np.random.choice(actions, p=probs)
                self.sweep_target = random.randint(1, 4) # 1 to 4 steps of momentum
                self.sweep_count = 0
                
            self.sweep_count += 1
            self.state = 'MOVE'
            self.state_start_time = now
                
        elapsed = now - self.state_start_time
        
        move_duration = 0.6 # FWD
        if self.current_action_id in [3, 4]:
            move_duration = 0.20
        elif self.current_action_id == 2:
            move_duration = 0.30
            
        stop_duration = 0.8
        
        if self.state == 'MOVE':
            if elapsed >= move_duration:
                self.state = 'STOP'
                self.state_start_time = now
                return (0, 0)
            else:
                if self.current_action_id == 1: return (130, 130)
                elif self.current_action_id == 2: return (-130, -130)
                elif self.current_action_id == 3: return (-110, 110)
                elif self.current_action_id == 4: return (110, -110)
                
        elif self.state == 'STOP':
            if elapsed >= stop_duration:
                self.state = 'WAITING'
                
            return (0, 0)
            
        return (0, 0)


class Markov:
    """
    Markov: Probabilistically selects actions and duration.
    Forward (p=0.6, 1000ms), Reverse (p=0.1, 500ms), Left (p=0.15, 200ms), Right (p=0.15, 200ms).
    Stops for 1000ms between actions.
    Reflex: Reverse until sensor_dist < 800, then STOP 1000ms, TURN 1-5 times, then resume.
    """
    def __init__(self):
        self.state = 'MOVE'
        self.state_start_time = time.time()
        self.current_action = self._sample_action()
        self.action_queue = []
        self.is_backing = False

    def _sample_action(self):
        actions = ['FWD', 'BACK', 'LEFT', 'RIGHT']
        probs = [0.6, 0.1, 0.15, 0.15]
        return np.random.choice(actions, p=probs)

    def get_action(self, sensor_dist):
        now = time.time()
        
        # Trigger Reflex
        if sensor_dist > AUTONOMY_THRESHOLD and not getattr(self, 'in_reflex', False):
            self.in_reflex = True
            self.action_queue = [] # clear any pending actions
            self.state = 'STOP'
            self.state_start_time = now
            
            turn_dir = random.choice(['LEFT', 'RIGHT'])
            self.action_queue.append('BACK') # single reversal first
            for _ in range(3):
                self.action_queue.append(turn_dir)

        if getattr(self, 'in_reflex', False) and len(self.action_queue) == 0:
            self.in_reflex = False
                
        # Normal State Machine
        elapsed = now - self.state_start_time
        move_duration = 0.6 # default FWD
        if self.current_action == 'LEFT':
            move_duration = 0.20
        elif self.current_action == 'RIGHT':
            move_duration = 0.20
        elif self.current_action == 'BACK':
            move_duration = 0.30
            
        stop_duration = 0.8
        
        if self.state == 'MOVE':
            if elapsed >= move_duration:
                self.state = 'STOP'
                self.state_start_time = now
        elif self.state == 'STOP':
            if elapsed >= stop_duration:
                self.state = 'MOVE'
                self.state_start_time = now
                
                if self.action_queue:
                    self.current_action = self.action_queue.pop(0)
                else:
                    self.current_action = self._sample_action()

        if self.state == 'MOVE':
            if self.current_action == 'FWD': return (130, 130)
            elif self.current_action == 'BACK': return (-130, -130)
            elif self.current_action == 'LEFT': return (-110, 110)
            elif self.current_action == 'RIGHT': return (110, -110)
            
        return (0, 0)


class MarkovTelemetry(Markov):
    """
    MarkovTelemetry: Functions exactly like Markov but with a different name 
    to trigger the 10-step telemetry warmup phase in the UI/Engine hook.
    """
    pass


class MarkovWASD:
    """
    MarkovWASD: A human-in-the-loop controller that enforces Markov durations and stop periods.
    Waits for a teleop action (WASD), then executes it for the Markov duration,
    followed by the mandatory 1000ms STOP, ignoring user input during execution.
    """
    def __init__(self):
        self.state = 'WAITING'
        self.state_start_time = 0
        self.current_action_id = 0
        self.is_backing = False
        self.action_queue = []

    def get_action(self, sensor_dist, teleop_action=0):
        # Cleanly convert to float if string
        try: dist = float(sensor_dist)
        except (ValueError, TypeError): dist = 0.0
        
        now = time.time()

        # [CRITICAL] Absolute priority for Intentional Stop (Goal Arrival)
        if teleop_action == 5:
            self.in_reflex = False
            self.action_queue = []
            # If not already executing the stop cycle, force an immediate transition
            if self.current_action_id != 5:
                self.state = 'WAITING'
        
        # Raw Safety Reflex: Natively handle proximity by injecting a macro-action turn sequence
        if dist > AUTONOMY_THRESHOLD and not getattr(self, 'in_reflex', False) and teleop_action != 5:
            self.in_reflex = True
            self.action_queue = [] # clear any pending actions
            self.state = 'STOP'
            self.state_start_time = now
            
            turn_dir = random.choice([3, 4]) # 3=LEFT, 4=RIGHT
            self.action_queue.append(2) # single reversal first
            
            if getattr(self, 'is_oracle_pacer', False):
                pass # Oracles map their own post-reversal navigation
            else:
                for _ in range(3): # Markov controllers need blind evasive turns
                    self.action_queue.append(turn_dir)

        if getattr(self, 'in_reflex', False) and len(self.action_queue) == 0:
            self.in_reflex = False
        
        # Determine duration based on the active action
        move_duration = 0.6 # default FWD (1)
        if self.current_action_id == 3 or self.current_action_id == 4: # LEFT or RIGHT
            move_duration = 0.20
        elif self.current_action_id == 2: # BACK
            move_duration = 0.30
        elif self.current_action_id == 5: # INTENTIONAL STOP
            move_duration = 0.20
            
        stop_duration = 0.8
        
        if self.state == 'WAITING':
            # Check if user pressed a valid WASD key (1=FWD, 2=REV, 3=LEFT, 4=RIGHT)
            teleop_src = teleop_action
            if self.action_queue:
                teleop_src = self.action_queue.pop(0)

            if teleop_src in [1, 2, 3, 4, 5]:
                self.state = 'MOVE'
                self.state_start_time = now
                self.current_action_id = teleop_src
            else:
                return (0, 0)
                
        # Calculate how long we've been in the CURRENT state
        elapsed = now - self.state_start_time
        
        if self.state == 'MOVE':
            if elapsed >= move_duration:
                self.state = 'STOP'
                self.state_start_time = now
                # We just transitioned to STOP. Send 0s immediately.
                return (0, 0)
            else:
                if self.current_action_id == 1: return (130, 130)
                elif self.current_action_id == 2: return (-130, -130)
                elif self.current_action_id == 3: return (-110, 110)
                elif self.current_action_id == 4: return (110, -110)
                elif self.current_action_id == 5: return (0, 0)
                
        elif self.state == 'STOP':
            if elapsed >= stop_duration:
                self.state = 'WAITING'
                self.current_action_id = 0
            
            # Important: return 0,0 during the STOP phase
            return (0, 0)
            
        return (0, 0)


class AlgorithmicOracle:
    """
    AlgorithmicOracle: Hardcoded trigonometric logic evaluated against true XYO telemetry.
    Re-mapped for continuous pacer transitions (simulating NN bounds).
    """
    def __init__(self):
        self.pacer = MarkovWASD()
        self.pacer.is_oracle_pacer = True
        
    @property
    def state(self):
        return self.pacer.state
        
    def get_action(self, sensor_dist=0, teleop_action=0, z_cur=None, z_goal=None, **kwargs):
        telemetry_cur = kwargs.get('telemetry_cur')
        telemetry_goal = kwargs.get('telemetry_goal')
        
        if teleop_action == 5:
            return self.pacer.get_action(sensor_dist, teleop_action=5)
        
        c_vec = telemetry_cur if telemetry_cur is not None else z_cur
        g_vec = telemetry_goal if telemetry_goal is not None else z_goal
        
        if c_vec is None or g_vec is None or len(c_vec) != 4 or len(g_vec) != 4:
            return self.pacer.get_action(sensor_dist, teleop_action=2) # 2 = BACK (Recovery action if tracking lost)
            
        import math
        cx, cy, cos_yaw, sin_yaw = c_vec[0], c_vec[1], c_vec[2], c_vec[3]
        gx, gy, gcos, gsin = g_vec[0], g_vec[1], g_vec[2], g_vec[3]
        
        yaw = math.atan2(sin_yaw, cos_yaw)
        g_yaw = math.atan2(gsin, gcos)
        
        dx = gx - cx
        dy = gy - cy
        e_dist = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        
        heading_err = (angle_to_goal - yaw + math.pi) % (2 * math.pi) - math.pi
        final_heading_err = (g_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        
        dist_thresh = 0.10
        angle_thresh = 0.3
        
        act_id = 0
        if e_dist < dist_thresh:
            if abs(final_heading_err) < angle_thresh: act_id = 5
            elif final_heading_err > 0: act_id = 4
            else: act_id = 3
        else:
            if abs(heading_err) < 0.4: act_id = 1
            elif abs(heading_err) > 2.7: act_id = 2
            elif heading_err > 0: act_id = 4
            else: act_id = 3
            
        return self.pacer.get_action(sensor_dist, teleop_action=act_id)
class ExplorationSystem:
    def __init__(self):
        self.current_algo = None
        self.markov = Markov()
        self.markov_wasd = MarkovWASD()
        self.markov_sweep = MarkovSweep()
        self.markov_telemetry = MarkovTelemetry()
        self.algo_oracle = AlgorithmicOracle()
        
    def set_algorithm(self, name):
        if name != self.current_algo:
            import logging
            logging.info(f"Autonomy Algorithm switched to: {name}")
            self.current_algo = name
            self.markov = Markov()
            self.markov_wasd = MarkovWASD()
            self.markov_sweep = MarkovSweep()
            self.markov_telemetry = MarkovTelemetry()
            self.algo_oracle = AlgorithmicOracle()

    def get_action(self, sensor_dist=0, teleop_action=0, z_cur=None, z_goal=None, img_cur=None, img_goal=None, latent_dist=None, **kwargs):
        try: dist = float(sensor_dist)
        except: dist = 0.0

        if self.current_algo == "Markov": return self.markov.get_action(dist)
        elif self.current_algo == "MarkovTelemetry": return self.markov_telemetry.get_action(dist)
        elif self.current_algo == "MarkovWASD": return self.markov_wasd.get_action(dist, teleop_action)
        elif self.current_algo == "MarkovSweep": return self.markov_sweep.get_action(dist)
        elif self.current_algo == "Algorithmic Oracle":
             if getattr(self, 'algo_oracle', None):
                 return self.algo_oracle.get_action(sensor_dist=dist, teleop_action=teleop_action, z_cur=z_cur, z_goal=z_goal, **kwargs)

        return (0, 0)
