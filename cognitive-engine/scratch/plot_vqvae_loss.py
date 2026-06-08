import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Parsed from training output
epochs = list(range(1, 51))
total_loss = [0.0572,0.0335,0.0255,0.0226,0.0202,0.0181,0.0168,0.0162,0.0158,0.0154,0.0152,0.0151,0.0150,0.0150,0.0153,0.0153,0.0152,0.0153,0.0154,0.0155,0.0155,0.0154,0.0155,0.0155,0.0155,0.0155,0.0156,0.0156,0.0156,0.0156,0.0157,0.0157,0.0156,0.0156,0.0156,0.0158,0.0158,0.0157,0.0157,0.0158,0.0159,0.0157,0.0156,0.0156,0.0158,0.0156,0.0156,0.0157,0.0158,0.0156]
recon_loss = [0.0539,0.0315,0.0249,0.0219,0.0196,0.0170,0.0155,0.0146,0.0141,0.0137,0.0134,0.0132,0.0130,0.0129,0.0129,0.0128,0.0127,0.0126,0.0126,0.0126,0.0125,0.0124,0.0124,0.0123,0.0123,0.0122,0.0122,0.0122,0.0122,0.0121,0.0122,0.0121,0.0120,0.0120,0.0119,0.0121,0.0120,0.0119,0.0119,0.0119,0.0119,0.0118,0.0118,0.0117,0.0118,0.0117,0.0117,0.0118,0.0118,0.0117]
vq_loss = [0.0033,0.0020,0.0006,0.0006,0.0006,0.0011,0.0012,0.0016,0.0017,0.0017,0.0018,0.0019,0.0020,0.0021,0.0024,0.0025,0.0026,0.0027,0.0028,0.0030,0.0030,0.0030,0.0031,0.0032,0.0032,0.0033,0.0034,0.0034,0.0034,0.0035,0.0035,0.0036,0.0036,0.0036,0.0036,0.0038,0.0038,0.0038,0.0038,0.0039,0.0039,0.0039,0.0038,0.0039,0.0040,0.0039,0.0039,0.0040,0.0040,0.0039]
perplexity = [1.49,12.27,51.24,58.43,96.03,91.51,99.19,100.01,101.85,103.16,103.03,103.03,103.38,104.62,103.99,103.73,102.60,103.86,102.64,103.03,101.39,101.46,102.89,104.47,102.29,102.68,101.84,101.96,102.56,102.61,102.31,102.89,102.46,102.50,103.36,100.91,102.81,101.51,103.27,102.33,101.55,99.57,100.55,102.57,101.73,101.68,102.65,102.39,101.56,102.22]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('VQ-VAE Training Results (512 codes × 32-dim, 4653 frames)', fontsize=15, fontweight='bold')

# 1. Total Loss
ax = axes[0, 0]
ax.plot(epochs, total_loss, color='#2563eb', linewidth=2)
ax.set_title('Total Loss', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.axvline(x=10, color='gray', linestyle='--', alpha=0.4, label='Convergence zone')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.065)

# 2. Recon vs VQ Loss (stacked area style)
ax = axes[0, 1]
ax.plot(epochs, recon_loss, color='#059669', linewidth=2, label='Reconstruction (MSE)')
ax.plot(epochs, vq_loss, color='#dc2626', linewidth=2, label='VQ Commitment')
ax.fill_between(epochs, recon_loss, alpha=0.15, color='#059669')
ax.fill_between(epochs, vq_loss, alpha=0.15, color='#dc2626')
ax.set_title('Loss Decomposition', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Perplexity
ax = axes[1, 0]
ax.plot(epochs, perplexity, color='#7c3aed', linewidth=2)
ax.axhline(y=512, color='gray', linestyle=':', alpha=0.5, label=f'Max (512 codes)')
ax.axhline(y=np.mean(perplexity[-10:]), color='#7c3aed', linestyle='--', alpha=0.5, 
           label=f'Final avg: {np.mean(perplexity[-10:]):.1f}')
ax.set_title('Codebook Perplexity (Effective Utilization)', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Perplexity')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 550)

# 4. VQ/Recon ratio
ratio = [v/r for v, r in zip(vq_loss, recon_loss)]
ax = axes[1, 1]
ax.plot(epochs, ratio, color='#ea580c', linewidth=2)
ax.set_title('VQ / Recon Loss Ratio', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Ratio')
ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.4, label='1:3 balance')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r'C:\Users\chris\.gemini\antigravity\brain\749bcf60-5c40-49d5-a3a5-f77ad57524fa\vqvae_training_plot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
