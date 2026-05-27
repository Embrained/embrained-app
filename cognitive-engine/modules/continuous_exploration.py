import time
import random
import logging
import numpy as np
from modules.exploration import AUTONOMY_THRESHOLD

def map_continuous_to_pwm(linear, angular):
    """
    Maps logical continuous commands [-1.0, 1.0] to hardware PWM values.
    Accounts for motor stall deadbands.
    """
    left = linear - angular  # Standard differential drive kinematic 
    right = linear + angular
    
    max_val = max(abs(left), abs(right), 1.0)
    left = left / max_val
    right = right / max_val
    
    def to_pwm(val):
        stall = 100
        max_pwm = 255
        if abs(val) < 0.05: return 0
        pwm = stall + (abs(val) * (max_pwm - stall))
        return int(pwm * (1 if val > 0 else -1))
        
    return to_pwm(left), to_pwm(right)


class ContinuousSweep:
    """
    Generates smooth, continuous motor trajectories.
    Interpolates current velocity towards a randomly sampled target velocity.
    """
    def __init__(self):
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.last_change_time = time.time()
        self.hold_duration = 0.0
        
    def get_action(self, sensor_dist):
        try: dist = float(sensor_dist)
        except: dist = 0.0
        
        now = time.time()
        
        # Soft Reflex: If object detected, blend into a sweeping reverse
        if dist > AUTONOMY_THRESHOLD:
            # We don't hard stop. We smoothly target reverse-turn.
            if self.target_linear >= 0:
                self.target_linear = random.uniform(-0.6, -0.3)
                self.target_angular = random.choice([-0.8, 0.8])
                self.last_change_time = now
                self.hold_duration = random.uniform(1.0, 1.5)
        else:
            if now - self.last_change_time > self.hold_duration:
                # Sample new smooth motion primitive
                # Bias forward movement, allow gentle sweeping turns
                self.target_linear = random.uniform(0.3, 1.0)
                
                # Sometime go straight, sometimes curve
                if random.random() > 0.5:
                    self.target_angular = 0.0
                else:
                    self.target_angular = random.uniform(-0.8, 0.8)
                    
                self.last_change_time = now
                self.hold_duration = random.uniform(0.5, 2.5)
                
        # Interpolate (Smooths out 10Hz commands)
        alpha = 0.2
        self.current_linear += (self.target_linear - self.current_linear) * alpha
        self.current_angular += (self.target_angular - self.current_angular) * alpha
        
        return map_continuous_to_pwm(self.current_linear, self.current_angular)


class ContinuousWASD:
    """
    Smooth teleoperation controller.
    """
    def __init__(self):
        self.current_linear = 0.0
        self.current_angular = 0.0
        
    def get_action(self, sensor_dist, teleop_action=0):
        target_linear = 0.0
        target_angular = 0.0
        
        # 1=FWD, 2=REV, 3=LEFT, 4=RIGHT, 5=STOP
        if teleop_action == 1:
            target_linear = 1.0
        elif teleop_action == 2:
            target_linear = -1.0
        elif teleop_action == 3:
            target_angular = -1.0 # In-place turn left
        elif teleop_action == 4:
            target_angular = 1.0  # In-place turn right
            
        try: dist = float(sensor_dist)
        except: dist = 0.0
        
        if dist > AUTONOMY_THRESHOLD and teleop_action != 5:
            # Optional: Add reflex override for teleop here if desired
            pass

        # Faster interpolation for teleop responsiveness
        alpha = 0.4
        self.current_linear += (target_linear - self.current_linear) * alpha
        self.current_angular += (target_angular - self.current_angular) * alpha
        
        if teleop_action == 5:
            self.current_linear = 0.0
            self.current_angular = 0.0
            return (0, 0)
            
        return map_continuous_to_pwm(self.current_linear, self.current_angular)


class ContinuousExplorationSystem:
    def __init__(self):
        self.current_algo = None
        self.algo_sweep = ContinuousSweep()
        self.algo_wasd = ContinuousWASD()
        
    def set_algorithm(self, name):
        if name != self.current_algo:
            logging.info(f"Continuous Autonomy Algorithm switched to: {name}")
            self.current_algo = name
            self.algo_sweep = ContinuousSweep()
            self.algo_wasd = ContinuousWASD()

    def get_action(self, sensor_dist=0, teleop_action=0, **kwargs):
        if self.current_algo == "ContinuousSweep": 
            return self.algo_sweep.get_action(sensor_dist)
        elif self.current_algo == "ContinuousWASD": 
            return self.algo_wasd.get_action(sensor_dist, teleop_action)

        return (0, 0)
