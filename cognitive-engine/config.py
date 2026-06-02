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

# --- NETWORK ---
ROBOT_IP = "192.168.4.1"
# UDP_PORT = 3000 # DEPRECATED: Switching to Legacy WS/HTTP
WS_URL = f"ws://{ROBOT_IP}/ws"
STREAM_URL = f"http://{ROBOT_IP}:81/stream"

HANDSHAKE_TIMEOUT = 2.0  # Seconds
LISTEN_TIMEOUT_MS = 2000 # Watchdog

# --- VISION ---
# --- VISION ---
IMG_W = 64
IMG_H = 64
IMG_CHANNELS = 3
RECORD_W = 320     # [NEW] Resolution for UI/Recording
RECORD_H = 240     # [NEW] Resolution for UI/Recording
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# --- MODEL ARCHITECTURE (ACT) ---
HIDDEN_DIM = 1024      
ACT_CHUNK_SIZE = 20    # 2.0s Horizon at 10Hz
ACT_TEMPORAL_AGG_K = 0.01
ACT_LATENT_DIM = 32
# [NEW] 64 (Latents) + 3 (v_pwm, w_pwm, sonar) = 67
INPUT_DIM = 64

# --- CONTROL ---
# 5-Byte Protocol: [LS, LD, RS, RD, FREQ]
# LD/RD: 1=Forward, 2=Reverse (Standard assumption, verify with firmware)
CMD_FWD_VAL = 1
CMD_REV_VAL = 2

# [REVERTED] 6-Class Discrete PWM Action Space
ACTION_DIM = 6

# Action ID to Semantics
ACTION_NAMES = {
    0: "STOP",
    1: "FWD",
    2: "REVERSE",
    3: "HARD_LEFT",
    4: "HARD_RIGHT",
    5: "INTENTIONAL_STOP"
}

# The explicit (Left, Right) PWM targets for each Discrete Action 0-4
ACTION_PWM_MAP = {
    0: (0, 0),        # Space
    1: (130, 130),    # W
    2: (-130, -130),  # S
    3: (-110, 110),   # A
    4: (110, -110),   # D
    5: (0, 0)         # Space (Intentional Stop)
}

# Motor Polarity & Tuning
# Legacy insight: Left motor might need inversion.
# If LEFT_MOTOR_INVERT is True, we swap FWD/REV values for the Left byte.
#
# !!! SPIKERBOT PROTOTYPE 2025 HARDWARE PATCH !!!
# This unit has non-standard wiring.
# - Forward logic (l:S, r:S) causes a RIGHT TURN.
# - Left logic (l:-S, r:S) causes FORWARD movement.
# - Right logic (l:S, r:-S) causes BACKWARD movement.
#
# teleop.py handles this with a custom map.
# comms.py relies on LEFT_MOTOR_INVERT=True to get Forward correct (l:-S, r:S),
# but Left/Right strafing might be reversed in autonomous mode unless accounted for.
# DO NOT CHANGE WITHOUT HARDWARE VERIFICATION.
LEFT_MOTOR_INVERT = True 
BASE_SPEED = 0.8       # Normalized speed (0.0 to 1.0)
TURN_SPEED = 0.8        # Normalized turn speed

# --- NAVIGATION ---
GOAL_SWITCH_INTERVAL = 20.0 # Seconds
CONTROL_FREQ = 10.0         # Hz
STOP_DISTANCE_THRESHOLD = 2.00
STOP_COOLDOWN_S = 0.2
GOAL_LED_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] # Green, Blue, Red
COLOR_NAME_MAP = {(0, 255, 0): "Green", (0, 0, 255): "Blue", (255, 0, 0): "Red"}

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
GOAL_DIR = os.path.join(DATA_DIR, 'goals')

