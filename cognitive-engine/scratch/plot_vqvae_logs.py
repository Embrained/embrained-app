import re
import matplotlib.pyplot as plt
import os

log_text = """
INFO:TrainVQVAE:Epoch 1/50 (71.5s) | Loss: 4.5288 | Recon: 0.0273 | InfoNCE: 0.6993 | VQ: 3.8022 | Perplexity: 2.21
INFO:TrainVQVAE:Epoch 2/50 (65.2s) | Loss: 0.7305 | Recon: 0.0210 | InfoNCE: 0.6931 | VQ: 0.0163 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 3/50 (63.7s) | Loss: 0.7163 | Recon: 0.0209 | InfoNCE: 0.6931 | VQ: 0.0023 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 4/50 (63.9s) | Loss: 0.7146 | Recon: 0.0209 | InfoNCE: 0.6931 | VQ: 0.0006 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 5/50 (63.6s) | Loss: 0.7145 | Recon: 0.0209 | InfoNCE: 0.6931 | VQ: 0.0005 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 6/50 (68.1s) | Loss: 0.7144 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0005 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 7/50 (63.1s) | Loss: 0.7144 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0004 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 8/50 (63.0s) | Loss: 0.7145 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0005 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 9/50 (63.3s) | Loss: 0.7145 | Recon: 0.0209 | InfoNCE: 0.6931 | VQ: 0.0005 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 10/50 (62.9s) | Loss: 0.7146 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0007 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 11/50 (63.1s) | Loss: 0.7149 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0009 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 12/50 (62.8s) | Loss: 0.7155 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0015 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 13/50 (64.6s) | Loss: 0.7186 | Recon: 0.0209 | InfoNCE: 0.6931 | VQ: 0.0046 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 14/50 (62.9s) | Loss: 0.7281 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0142 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 15/50 (62.3s) | Loss: 0.7222 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0082 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 16/50 (62.5s) | Loss: 0.7510 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0371 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 17/50 (62.9s) | Loss: 0.7265 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0125 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 18/50 (62.6s) | Loss: 0.7372 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0233 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 19/50 (62.8s) | Loss: 0.7258 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0119 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 20/50 (63.2s) | Loss: 0.7478 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0338 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 21/50 (62.9s) | Loss: 0.7451 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0312 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 22/50 (61.9s) | Loss: 0.7174 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0034 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 23/50 (63.3s) | Loss: 0.7330 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0190 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 24/50 (63.5s) | Loss: 0.7413 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0274 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 25/50 (62.3s) | Loss: 0.7402 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0262 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 26/50 (61.9s) | Loss: 0.7240 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0101 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 27/50 (62.6s) | Loss: 0.7545 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0405 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 28/50 (62.6s) | Loss: 0.7169 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0029 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 29/50 (62.6s) | Loss: 0.7362 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0222 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 30/50 (62.2s) | Loss: 0.7326 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0186 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 31/50 (62.6s) | Loss: 0.7304 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0164 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 32/50 (63.5s) | Loss: 0.7387 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0248 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 33/50 (63.4s) | Loss: 0.7382 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0243 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 34/50 (62.3s) | Loss: 0.7337 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0198 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 35/50 (63.0s) | Loss: 0.7235 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0095 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 36/50 (62.8s) | Loss: 0.7372 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0232 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 37/50 (64.0s) | Loss: 0.7330 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0190 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 38/50 (63.2s) | Loss: 0.7229 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0090 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 39/50 (63.1s) | Loss: 0.7384 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0244 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 40/50 (61.7s) | Loss: 0.7276 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0137 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 41/50 (62.9s) | Loss: 0.7272 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0133 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 42/50 (63.1s) | Loss: 0.7374 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0234 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 43/50 (63.6s) | Loss: 0.7315 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0175 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 44/50 (75.7s) | Loss: 0.7206 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0066 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 45/50 (70.7s) | Loss: 0.7389 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0249 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 46/50 (63.7s) | Loss: 0.7261 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0121 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 47/50 (63.6s) | Loss: 0.7310 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0170 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 48/50 (64.2s) | Loss: 0.7335 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0195 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 49/50 (64.0s) | Loss: 0.7257 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0118 | Perplexity: 1.00
INFO:TrainVQVAE:Epoch 50/50 (65.0s) | Loss: 0.7332 | Recon: 0.0208 | InfoNCE: 0.6931 | VQ: 0.0192 | Perplexity: 1.00
"""

epochs = []
losses = []
recons = []
infonces = []
vqs = []
perps = []

for line in log_text.strip().split('\n'):
    match = re.search(r'Epoch (\d+)/.*?Loss: ([\d.]+) \| Recon: ([\d.]+) \| InfoNCE: ([\d.]+) \| VQ: ([\d.]+) \| Perplexity: ([\d.]+)', line)
    if match:
        epochs.append(int(match.group(1)))
        losses.append(float(match.group(2)))
        recons.append(float(match.group(3)))
        infonces.append(float(match.group(4)))
        vqs.append(float(match.group(5)))
        perps.append(float(match.group(6)))

out_dir = r"c:\Users\chris\Embrained\software_suite\data"
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Total Loss', color='black', linewidth=2)
plt.plot(epochs, recons, label='Recon Loss', color='blue')
plt.plot(epochs, infonces, label='InfoNCE Loss', color='green')
plt.plot(epochs, vqs, label='VQ Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'vqvae_training_losses.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs, perps, label='Perplexity (Active Codes)', color='purple', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Codebook Perplexity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'vqvae_training_perplexity.png'))
plt.close()

print("Plots saved successfully.")
