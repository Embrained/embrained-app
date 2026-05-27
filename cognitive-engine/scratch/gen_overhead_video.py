"""
Generate overhead tracking video from webcam frames + telemetry.
Outputs: website/video/overhead_tracking.mp4
"""
import cv2
import numpy as np
import pandas as pd
import os, glob, sys

# --- Config ---
SESSION = "markov_2026-04-15_19-08-52"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SESSION_DIR = os.path.join(DATA_DIR, SESSION)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "website", "video")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "overhead_tracking.mp4")
FPS = 4

# Action map
ACTION_NAMES = {
    (130, 130): ("FWD", (0, 200, 0)),
    (-130, -130): ("REV", (0, 0, 200)),
    (-110, 110): ("SPIN-L", (200, 200, 0)),
    (110, -110): ("SPIN-R", (200, 0, 200)),
    (0, 0): ("STOP", (128, 128, 128)),
}

def get_action_label(left, right):
    key = (int(left), int(right))
    if key in ACTION_NAMES:
        return ACTION_NAMES[key]
    if left > 0 and right > 0:
        return ("FWD", (0, 200, 0))
    if left < 0 and right < 0:
        return ("REV", (0, 0, 200))
    if left < 0 and right > 0:
        return ("SPIN-L", (200, 200, 0))
    if left > 0 and right < 0:
        return ("SPIN-R", (200, 0, 200))
    return ("STOP", (128, 128, 128))

# --- Load episode CSV ---
csv_path = os.path.join(SESSION_DIR, "episode_data.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} frames from {SESSION}")

# --- Load telemetry for this session ---
telem_path = os.path.join(DATA_DIR, "master_telemetry.csv")
telem = pd.read_csv(telem_path)
# Filter to this session
telem_session = telem[telem['img_dir'].str.contains(SESSION)].copy()
telem_session = telem_session.sort_values('ts').reset_index(drop=True)
print(f"Telemetry rows for session: {len(telem_session)}")

# Build ts->telemetry lookup
telem_dict = {}
for _, row in telem_session.iterrows():
    telem_dict[int(row['ts'])] = row

# --- Generate video ---
# Get first webcam frame to determine size
first_wc = os.path.join(SESSION_DIR, "images", f"webcam_{df.iloc[0]['image_file']}")
sample = cv2.imread(first_wc)
if sample is None:
    print(f"ERROR: Cannot read {first_wc}")
    sys.exit(1)

H, W = sample.shape[:2]
print(f"Frame size: {W}x{H}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

trail = []  # List of (cx, cy) for trail
MAX_TRAIL = 80

for i, row in df.iterrows():
    wc_path = os.path.join(SESSION_DIR, "images", f"webcam_{row['image_file']}")
    frame = cv2.imread(wc_path)
    if frame is None:
        continue

    # Get telemetry for this timestamp
    ts_key = int(row['timestamp'] * 1000)  # episode CSV has seconds, telemetry has ms
    ts_ms = str(ts_key)
    # Find closest telemetry row
    trow = None
    # Try exact match from image filename
    fname_ts = row['image_file'].replace('frame_', '').replace('.jpg', '')
    if int(fname_ts) in telem_dict:
        trow = telem_dict[int(fname_ts)]
    
    left_cmd = int(row['pwm_left'])
    right_cmd = int(row['pwm_right'])
    ir = float(row['ir_reading'])
    action_name, action_color = get_action_label(left_cmd, right_cmd)

    if trow is not None:
        cx, cy = int(trow['cx']), int(trow['cy'])
        yaw = float(trow['yaw_deg'])
        
        trail.append((cx, cy))
        if len(trail) > MAX_TRAIL:
            trail = trail[-MAX_TRAIL:]
        
        # Draw trail with fade
        for j in range(1, len(trail)):
            alpha = j / len(trail)
            color = (int(100 * alpha), int(220 * alpha), int(255 * alpha))
            thickness = max(1, int(2 * alpha))
            cv2.line(frame, trail[j-1], trail[j], color, thickness, cv2.LINE_AA)
        
        # Draw heading arrow
        arrow_len = 25
        yaw_rad = np.radians(yaw)
        dx = int(arrow_len * np.cos(yaw_rad))
        dy = -int(arrow_len * np.sin(yaw_rad))  # OpenCV y is flipped
        cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
        
        # Draw position dot
        cv2.circle(frame, (cx, cy), 5, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # --- HUD Overlay ---
    # Semi-transparent bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - 40), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Action label
    cv2.putText(frame, action_name, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_color, 2, cv2.LINE_AA)
    
    # IR bar
    ir_normalized = min(ir / 2000.0, 1.0)
    bar_w = int(120 * ir_normalized)
    bar_color = (0, 255, 0) if ir < 800 else ((0, 200, 255) if ir < 1500 else (0, 0, 255))
    cv2.rectangle(frame, (W - 140, H - 30), (W - 140 + bar_w, H - 14), bar_color, -1)
    cv2.rectangle(frame, (W - 140, H - 30), (W - 20, H - 14), (200, 200, 200), 1)
    cv2.putText(frame, "IR", (W - 160, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Frame counter
    cv2.putText(frame, f"t={i}", (W // 2 - 20, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    writer.write(frame)

writer.release()
print(f"Written {OUT_PATH} ({len(df)} frames at {FPS} fps)")
