# Embrained

**Train and deploy neural navigation agents on low-cost robots — locally, privately, on your own PC.**

Embrained offloads all heavy compute from the robot to your local GPU over WiFi. You collect real-world data by driving the robot around your home, train vision encoders and navigation policies on your machine, then deploy them back to the robot for autonomous goal-directed navigation. No cloud. No expensive onboard hardware.

## Quick Start

### 1. Install

```bash
git clone https://github.com/Embrained/embrained-app.git
cd embrained-app
```

**Windows:** `setup.bat` · **Mac/Linux:** `./setup.sh`

This creates an isolated virtual environment with Python 3.12, PyTorch, and Node.js.

### 2. Launch

**Windows:** `start.bat` · **Mac/Linux:** `./start.sh`

Open [http://localhost:8080](http://localhost:8080) to access the dashboard for teleoperation, live telemetry, and autonomy control.

### 3. Train

Activate your virtual environment first (`venv\Scripts\activate.bat` on Windows, `source venv/bin/activate` on Mac/Linux), then run from the project root:

```bash
# 1. Consolidate raw transition data into a training dataset
python cognitive-engine/scripts/prepare_dataset.py

# 2. Train the VQ-VAE encoder (compresses camera frames into discrete latents)
python cognitive-engine/backend/training/train_vqvae.py

# 3. Train the CQL navigation policy (maps latents to motor commands)
python cognitive-engine/backend/training/train_discrete_fixed_goal_cql.py
```

### 4. Deploy

1. Power on your Plexus robot and verify its WiFi connection in the app.
2. Open the **Autonomy Panel**, select your trained models, and click **Start Autonomy**.

The robot navigates to goal locations in real time, running entirely on your local hardware.

---

## Optional Setup

**CUDA (GPU acceleration):**
The default install is CPU-only. For faster training with an NVIDIA GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Simulation (PyBullet):**
Requires a C/C++ compiler ([MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) on Windows):

```bash
pip install -r requirements-simulation.txt
```

---

## Community

Share your trained weights, datasets, and demo videos on our [Discord](https://discord.com/channels/1487132795833684228/1487132796920004640). Contributed model weights feed into a federated **Model Soup** — enabling generalization across diverse home environments without sharing raw data.

## License

- **Software** — [GPLv3](LICENSE). Open to audit, modify, and contribute.
- **Your locally-trained weights** — 100% yours, no restrictions.
- **Pre-trained / aggregated weights** distributed by Embrained — governed by the Embrained Open-Weights EULA (non-commercial use permitted; commercial use requires a license). See [LICENSE](LICENSE) for full terms.
