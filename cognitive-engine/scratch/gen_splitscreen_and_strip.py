"""
Generate split-screen POV+Overhead video and forward sequence strip.
Outputs: website/video/splitscreen_pov.mp4, website/images/forward_sequence.png
"""
import cv2
import numpy as np
import pandas as pd
import os, sys

SESSION = "markov_2026-04-15_19-08-52"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SESSION_DIR = os.path.join(DATA_DIR, SESSION)
VID_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "website", "video")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "website", "images")
os.makedirs(VID_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
FPS = 4

ACTION_NAMES = {
    (130, 130): "FWD", (-130, -130): "REV",
    (-110, 110): "SPIN-L", (110, -110): "SPIN-R",
}

def get_action(l, r):
    key = (int(l), int(r))
    if key in ACTION_NAMES: return ACTION_NAMES[key]
    if l > 0 and r > 0: return "FWD"
    if l < 0 and r < 0: return "REV"
    if l < 0 and r > 0: return "SPIN-L"
    if l > 0 and r < 0: return "SPIN-R"
    return "STOP"

# Load data
df = pd.read_csv(os.path.join(SESSION_DIR, "episode_data.csv"))
telem = pd.read_csv(os.path.join(DATA_DIR, "master_telemetry.csv"))
telem_s = telem[telem['img_dir'].str.contains(SESSION)].sort_values('ts').reset_index(drop=True)
telem_dict = {int(r['ts']): r for _, r in telem_s.iterrows()}
print(f"Loaded {len(df)} frames")

# ========== SPLIT-SCREEN VIDEO ==========
OUT_W, OUT_H = 960, 480  # side-by-side
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(os.path.join(VID_DIR, "splitscreen_pov.mp4"), fourcc, FPS, (OUT_W, OUT_H))

# Use frames 30-110 (a nice exploration segment)
start_i, end_i = 30, min(110, len(df))
trail = []

for i in range(start_i, end_i):
    row = df.iloc[i]
    pov_path = os.path.join(SESSION_DIR, "images", row['image_file'])
    wc_path = os.path.join(SESSION_DIR, "images", f"webcam_{row['image_file']}")
    
    pov = cv2.imread(pov_path)
    wc = cv2.imread(wc_path)
    if pov is None or wc is None:
        continue
    
    # Resize POV to fill left half
    pov_big = cv2.resize(pov, (OUT_W // 2, OUT_H))
    wc_fit = cv2.resize(wc, (OUT_W // 2, OUT_H))
    
    # Add telemetry overlay to webcam
    fname_ts = row['image_file'].replace('frame_', '').replace('.jpg', '')
    trow = telem_dict.get(int(fname_ts))
    
    if trow is not None:
        # Scale cx,cy from 640x480 to half-width
        sx = (OUT_W // 2) / 640.0
        sy = OUT_H / 480.0
        cx, cy = int(trow['cx'] * sx), int(trow['cy'] * sy)
        yaw = float(trow['yaw_deg'])
        
        trail.append((cx, cy))
        if len(trail) > 60:
            trail = trail[-60:]
        
        for j in range(1, len(trail)):
            a = j / len(trail)
            cv2.line(wc_fit, trail[j-1], trail[j], (int(80*a), int(200*a), int(255*a)), max(1, int(2*a)), cv2.LINE_AA)
        
        cv2.circle(wc_fit, (cx, cy), 6, (0, 180, 255), -1, cv2.LINE_AA)
        arrow_len = 30
        dx = int(arrow_len * np.cos(np.radians(yaw)))
        dy = -int(arrow_len * np.sin(np.radians(yaw)))
        cv2.arrowedLine(wc_fit, (cx, cy), (cx+dx, cy+dy), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
    
    # Labels
    action = get_action(row['pwm_left'], row['pwm_right'])
    cv2.putText(pov_big, "ROBOT POV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pov_big, action, (10, OUT_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(wc_fit, "OVERHEAD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    combined = np.hstack([pov_big, wc_fit])
    # Separator line
    cv2.line(combined, (OUT_W // 2, 0), (OUT_W // 2, OUT_H), (255, 255, 255), 2)
    
    writer.write(combined)

writer.release()
print(f"Split-screen video: {end_i - start_i} frames")

# ========== FORWARD SEQUENCE STRIP ==========
# Find longest run of consecutive forward actions
df['action'] = df.apply(lambda r: get_action(r['pwm_left'], r['pwm_right']), axis=1)
fwd_mask = df['action'] == 'FWD'

# Find runs
best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
for i, is_fwd in enumerate(fwd_mask):
    if is_fwd:
        if cur_len == 0:
            cur_start = i
        cur_len += 1
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len
    else:
        cur_len = 0

print(f"Longest forward run: {best_len} frames starting at {best_start}")

# Take up to 8 frames
n_show = min(8, best_len)
strip_frames = []
for i in range(best_start, best_start + n_show):
    row = df.iloc[i]
    pov = cv2.imread(os.path.join(SESSION_DIR, "images", row['image_file']))
    wc = cv2.imread(os.path.join(SESSION_DIR, "images", f"webcam_{row['image_file']}"))
    if pov is None or wc is None:
        continue
    
    # Resize POV to 160x120 and webcam to 160x120
    pov_s = cv2.resize(pov, (160, 120))
    wc_s = cv2.resize(wc, (160, 120))
    
    # Stack vertically: POV on top, overhead below
    cell = np.vstack([pov_s, wc_s])
    
    # Add step number
    cv2.putText(cell, f"t={i - best_start}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cell, "FWD", (5, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    
    strip_frames.append(cell)

if strip_frames:
    strip = np.hstack(strip_frames)
    out_path = os.path.join(IMG_DIR, "forward_sequence.png")
    cv2.imwrite(out_path, strip)
    print(f"Forward strip: {out_path} ({strip.shape[1]}x{strip.shape[0]})")
