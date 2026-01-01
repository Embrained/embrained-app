
import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import math
import logging
from config import ACTION_NAMES, CMD_FWD_VAL, CMD_REV_VAL

class Simulator:

    def __init__(self, headless=True):
        self.headless = headless
        self.client_id = p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Load Plane
        self.planeId = p.loadURDF("plane.urdf")
        
        # Load Robot
        startPos = [0, 0, 0.5]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robotId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
        
        # Internal State
        self.img_w = 160
        self.img_h = 120
        self.fov = 60
        self.aspect = self.img_w / self.img_h
        self.near = 0.02
        self.far = 10
        
        # Camera Offset
        self.cam_local_pos = [0, 0, 0.4] 
        self.cam_local_orn = p.getQuaternionFromEuler([0, 0, 0]) 
        
        # Environment
        self._setup_environment()
        
        self.running = True
        self.latest_frame = None
        self.telemetry = {'dist': '0', 'batt': '100'}
        
        # Kinematic State
        self.linear_vel = 0.0 # m/s (Local X)
        self.angular_vel = 0.0 # rad/s (Yaw)
        self.last_step_time = time.time()
        
        logging.info("Simulator Initialized (PyBullet: Kinematic Mode)")

    def _setup_environment(self):
        # Add some random boxes
        for i in range(5):
            p.loadURDF("cube.urdf", [2 + i, i*0.5, 0.5], globalScaling=0.5)

    def get_latest_frame(self):
        """
        Synthesize image and apply Kinematic Updates.
        """
        # --- Time Step ---
        now = time.time()
        dt = now - self.last_step_time
        self.last_step_time = now
        
        # --- Kinematic Update (Bypassing Physics) ---
        if self.linear_vel != 0 or self.angular_vel != 0:
            pos, orn = p.getBasePositionAndOrientation(self.robotId)
            euler = p.getEulerFromQuaternion(orn)
            yaw = euler[2]
            
            # Update Yaw
            new_yaw = yaw + (self.angular_vel * dt)
            
            # Update Position (Move along NEW yaw)
            # Linear velocity is along local X
            dx = self.linear_vel * math.cos(new_yaw) * dt
            dy = self.linear_vel * math.sin(new_yaw) * dt
            
            new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]
            new_orn = p.getQuaternionFromEuler([euler[0], euler[1], new_yaw])
            
            # Force Reset (Teleport) to new state
            p.resetBasePositionAndOrientation(self.robotId, new_pos, new_orn)

        # --- Rendering ---
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_pos_world = np.array(pos) + rot_mat.dot(np.array(self.cam_local_pos))
        
        look_dir_local = np.array([1, 0, 0]) 
        look_dir_world = rot_mat.dot(look_dir_local)
        target_pos = cam_pos_world + look_dir_world
        
        view_matrix = p.computeViewMatrix(cam_pos_world, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        
        w, h, rgb, depth, seg = p.getCameraImage(
            self.img_w, self.img_h, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER
        )
        
        img = np.array(rgb, dtype=np.uint8).reshape((self.img_h, self.img_w, 4))
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        self.latest_frame = img
        return img

    def send_command(self, action_id):
        """
        Set kinematic target velocities.
        """
        # Tuning
        MOVE_SPEED = 2.0  # m/s
        TURN_SPEED = 4.0  # rad/s (~230 deg/s)
        
        # Map
        # 0: FWD
        # 1: LEFT
        # 2: RIGHT
        # 3: STOP
        # 4: BACK
        
        if action_id == 0: # FWD
            self.linear_vel = MOVE_SPEED
            self.angular_vel = 0
        elif action_id == 1: # LEFT
            self.linear_vel = 0
            self.angular_vel = TURN_SPEED # Positive yaw is usually Left (CCW)
        elif action_id == 2: # RIGHT
            self.linear_vel = 0
            self.angular_vel = -TURN_SPEED
        elif action_id == 3: # STOP
            self.linear_vel = 0
            self.angular_vel = 0
        elif action_id == 4: # BACK
            self.linear_vel = -MOVE_SPEED
            self.angular_vel = 0
            
        # Dummy return to satisfy comms interface
        return f"k:{self.linear_vel:.2f},{self.angular_vel:.2f};"

    def set_led(self, rgb_tuple):
        return f"d:{rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]};"

    def send_sound_command(self, freq):
        return f"s:{freq};"

    def close(self):
        p.disconnect()
