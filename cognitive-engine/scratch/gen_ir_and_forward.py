"""
Generate IR reflex montage: frames where IR spikes (obstacle) followed by reversal.
Also: find best forward runs across ALL sessions.
Outputs: website/video/ir_reflex_montage.mp4, website/images/forward_sequence.png (updated)
"""
import cv2
import numpy as np
import pandas as pd
import json
import os, sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VID_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "website", "video")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "website", "images")
os.makedirs(VID_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Load all transitions
with open(os.path.join(DATA_DIR, "all_transitions.json")) as f:
    transitions = json.load(f)
print(f"Loaded {len(transitions)} transitions")

# ========== IR REFLEX MONTAGE ==========
# Find frames where IR > 1500 (close obstacle) and next action is reverse (macro_action=2)
ir_events = []
for i in range(len(transitions) - 1):
    t = transitions[i]
    t_next = transitions[i + 1]
    ir = t.get('dist', 0)
    if ir > 1500 and t_next.get('macro_action') == 2 and t['session'] == t_next['session']:
        ir_events.append((i, ir, t['image_path'], t_next['image_path'], t['session']))

print(f"Found {len(ir_events)} IR reflex events")

# Pick up to 20 best events (highest IR = closest obstacle)
ir_events.sort(key=lambda x: -x[1])
ir_events = ir_events[:20]

# Build montage video: show before frame, then after frame, repeat
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
OUT_W, OUT_H = 640, 240
writer = cv2.VideoWriter(os.path.join(VID_DIR, "ir_reflex_montage.mp4"), fourcc, 3, (OUT_W, OUT_H))

for idx, (i, ir_val, img_before, img_after, session) in enumerate(ir_events):
    before_path = os.path.join(DATA_DIR, img_before)
    after_path = os.path.join(DATA_DIR, img_after)
    
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)
    if before is None or after is None:
        continue
    
    # Resize both to half width
    bh = cv2.resize(before, (OUT_W // 2, OUT_H))
    ah = cv2.resize(after, (OUT_W // 2, OUT_H))
    
    # Red border on "before" (danger), green on "after" (reversed)
    cv2.rectangle(bh, (0, 0), (OUT_W // 2 - 1, OUT_H - 1), (0, 0, 255), 3)
    cv2.rectangle(ah, (0, 0), (OUT_W // 2 - 1, OUT_H - 1), (0, 200, 0), 3)
    
    # Labels
    cv2.putText(bh, f"OBSTACLE (IR={int(ir_val)})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(ah, "REVERSE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(bh, f"#{idx+1}", (OUT_W // 2 - 40, OUT_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    combined = np.hstack([bh, ah])
    cv2.line(combined, (OUT_W // 2, 0), (OUT_W // 2, OUT_H), (255, 255, 255), 2)
    
    # Hold each pair for 1 second (3 frames at 3fps)
    for _ in range(3):
        writer.write(combined)

writer.release()
print(f"IR reflex montage: {len(ir_events)} events")

# ========== BETTER FORWARD SEQUENCE ==========
# Find longest forward run across all transitions
best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
for i, t in enumerate(transitions):
    if t['macro_action'] == 1:
        if cur_len == 0: cur_start = i
        cur_len += 1
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len
    else:
        cur_len = 0

print(f"Best forward run across all data: {best_len} frames starting at {best_start}")
if best_len >= 5:
    session = transitions[best_start]['session']
    print(f"  Session: {session}")

# Take 8 frames
n_show = min(8, best_len)
strip_frames = []
for i in range(best_start, best_start + n_show):
    t = transitions[i]
    img_path = os.path.join(DATA_DIR, t['image_path'])
    pov = cv2.imread(img_path)
    if pov is None:
        continue
    
    pov_s = cv2.resize(pov, (160, 120))
    
    # Add step label and IR
    cv2.putText(pov_s, f"t={i - best_start}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    ir = t.get('dist', 0)
    cv2.putText(pov_s, f"IR:{int(ir)}", (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(pov_s, "FWD", (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    
    strip_frames.append(pov_s)

if strip_frames:
    strip = np.hstack(strip_frames)
    out_path = os.path.join(IMG_DIR, "forward_sequence.png")
    cv2.imwrite(out_path, strip)
    print(f"Forward strip: {out_path} ({strip.shape[1]}x{strip.shape[0]}, {n_show} frames)")
