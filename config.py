
import os

# --- NETWORK ---
# --- NETWORK ---
ROBOT_IP = "192.168.4.1"
# UDP_PORT = 3000 # DEPRECATED: Swithing to Legacy WS/HTTP
WS_URL = f"ws://{ROBOT_IP}/ws"
STREAM_URL = f"http://{ROBOT_IP}:81/stream"

HANDSHAKE_TIMEOUT = 2.0  # Seconds
LISTEN_TIMEOUT_MS = 2000 # Watchdog

# --- VISION ---
IMG_W = 160
IMG_H = 120
IMG_CHANNELS = 3
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# --- CONTROL ---
# 5-Byte Protocol: [LS, LD, RS, RD, FREQ]
# LD/RD: 1=Forward, 2=Reverse (Standard assumption, verify with firmware)
CMD_FWD_VAL = 1
CMD_REV_VAL = 2

# Action ID to Semantics
# 0: FWD, 1: LEFT, 2: RIGHT, 3: STOP
ACTION_NAMES = {
    0: "FORWARD",
    1: "LEFT",
    2: "RIGHT",
    3: "STOP",
    4: "BACKWARD"
}

# Motor Polarity & Tuning
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
BASE_SPEED = 100       # Standard operating speed (0-255)
TURN_SPEED = 80        # Speed when turning

# --- NAVIGATION ---
GOAL_SWITCH_INTERVAL = 15.0 # Seconds
CONTROL_FREQ = 10.0         # Hz
STOP_DISTANCE_THRESHOLD = 2.0
STOP_COOLDOWN_S = 0.5
GOAL_LED_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] # Green, Blue, Red
COLOR_NAME_MAP = {(0, 255, 0): "Green", (0, 0, 255): "Blue", (255, 0, 0): "Red"}

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')

# Ensure logging dirs exist
os.makedirs(LOGS_DIR, exist_ok=True)
