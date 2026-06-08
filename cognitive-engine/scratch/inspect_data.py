import os, csv

root = r'c:\Users\chris\Embrained\embrained-app\data'
dirs = sorted(os.listdir(root))

print("Recording Statistics:")
print("=" * 80)
total_frames = 0

for d in dirs:
    dp = os.path.join(root, d)
    if not os.path.isdir(dp):
        continue
    csv_path = os.path.join(dp, "episode_data.csv")
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    n = len(rows) - 1  # minus header
    total_frames += n
    
    t0 = float(rows[1][0])
    t1 = float(rows[-1][0])
    duration = t1 - t0
    rate = n / max(1, duration)
    
    print(f"  {d}: {n:>5} frames | {duration:>6.0f}s ({duration/60:.1f}min) | {rate:.1f} Hz")

print(f"\n  TOTAL: {total_frames} frames")
print(f"  Image resolution: 320x240 RGB (downscaled to 64x64 for training)")
