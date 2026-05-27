import matplotlib.pyplot as plt
import os

losses = [
    586573.5154, 274126.8725, 265702.7303, 261811.3358, 258928.6875,
    256740.0945, 254632.3654, 253079.4695, 251813.4464, 250627.6016,
    249335.1698, 248580.5730, 246896.8754, 245773.4089, 244588.3829,
    243658.8852, 243308.0312, 241702.0018, 240198.3546, 239103.0941,
    238048.3037, 236658.9257, 235017.1654, 234053.8407, 232791.6132,
    231447.2121, 230276.3386, 229507.4680, 228808.4652, 227640.7550
]
epochs = list(range(1, 31))

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=5)
plt.title("IR Reflex Forward Model Training Loss\nModel: tinyvae-vae_continuous_20260418_145332-hello_world_ir_reflex.pth", fontsize=12)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save locally to /data
save_path_data = os.path.abspath(r"data\ir_reflex_loss.png")
plt.savefig(save_path_data, dpi=150)
print(f"Saved to {save_path_data}")

# Save explicitly to the Antigravity artifacts directory so it renders in the chat
artifacts_path = os.path.abspath(r"C:\Users\chris\.gemini\antigravity\brain\579c23c7-abeb-4e15-acc5-06311e12eb85\ir_reflex_loss.png")
os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
plt.savefig(artifacts_path, dpi=150)
print(f"Saved to {artifacts_path}")
