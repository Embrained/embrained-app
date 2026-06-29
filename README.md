# Embrained

**Train and deploy neural navigation agents on low-cost robots - locally, privately, on your own PC.**

Embrained offloads all heavy compute from the robot to your local GPU over WiFi. You collect real-world data by driving the robot around your home, train vision encoders and navigation policies on your machine, then deploy them back to the robot for autonomous goal-directed navigation. No cloud. No expensive onboard hardware.

## Quick Start

### 1. Install

**Step 1.** Install [Node.js](https://nodejs.org/) (required for the frontend dashboard).

**Step 2.** Clone the repository:

```bash
git clone https://github.com/Embrained/embrained-app.git
cd embrained-app
```

### 2. Connect to Robot

1. Power on your Plexus robot. 
2. Wait for the tally light to blink at 1 Hz, indicating it is ready to connect.
3. Connect your computer to the robot's WiFi access point (SSID typically looks like `Plexus_XXXX`).
4. The tally light will change to steady ON when your computer has successfully connected to the robot.

### 3. Launch

Run the setup script to create an isolated virtual environment with Python 3.12, PyTorch, and other dependencies, then launch the app:

- **Windows:** run `setup.bat`, then run `start.bat`
- **Mac/Linux:** run `./setup.sh`, then run `./start.sh`

Open [http://localhost:8080](http://localhost:8080) to access the dashboard for teleoperation, live telemetry, and autonomy control.

### 4. Collect Data

Use the dashboard to drive the robot and record training data:

1. Drive the robot around using **manual control** (keyboard/gamepad) or start an **autonomous controller** to explore automatically.
2. Press the **REC** button to begin recording. The robot saves camera frames, motor commands, and sensor readings as it moves.
3. Press **REC** again to stop. Each recording session is saved to `data/markov_<timestamp>/` inside the project directory.

Record several sessions covering your environment from different starting positions. More diverse data → better navigation.

### 5. Train

Activate your virtual environment first (`venv\Scripts\activate.bat` on Windows, `source venv/bin/activate` on Mac/Linux), then run from the project root.

There are two training options:

#### Option A - Fixed-Goal Navigation (Simplest)

Trains a policy that drives the robot to **one specific location** from anywhere. This is the quickest way to get a working autonomous agent.

**Step 1.** Consolidate raw recordings into a training dataset:

```bash
python cognitive-engine/scripts/prepare_dataset.py
```

**Step 2.** Create a `data/goals/` folder and copy **5-30 camera frames** that show your desired goal location (the place the robot should drive to). These should be images the robot captured while at or near the goal - you can find them inside your recording folders under `data/markov_<timestamp>/images/`.

```bash
mkdir data/goals
# Copy goal images into data/goals/  (e.g. .jpg or .png frames showing the target location)
```

**Step 3.** Train the Contrastive Visuomotor Encoder (CVE) to compress camera frames into latent state vectors:

```bash
python cognitive-engine/backend/training/train_cve.py
```

**Step 4.** Train the fixed-goal CQL navigation policy:

```bash
python cognitive-engine/backend/training/train_cve_cql_fixed_goal.py
```

The script reads your goal images from `data/goals/`, uses the CVE to automatically detect high-precision terminal states via nearest-neighbor clustering, and trains an offline RL policy using CVE representations that maps observations to motor commands to drive toward the target.

#### Option B - Goal-Conditioned Navigation (Any Goal ↔ Any Start)

Trains a policy that can navigate to **any goal from any location**. Once deployed, you can select goals interactively through the **Latent Space** panel in the dashboard.

**Step 1.** Consolidate raw recordings into a training dataset (skip if already done):

```bash
python cognitive-engine/scripts/prepare_dataset.py
```

**Step 2.** Train the CVE:

```bash
python cognitive-engine/backend/training/train_cve.py
```

**Step 3.** Train the goal-conditioned CQL policy using Hindsight Experience Replay:

```bash
python cognitive-engine/backend/training/train_cve_cql_goal_conditioned.py
```

This uses [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495) to relabel every trajectory with multiple future frames as goals, producing a policy that generalizes across all locations in your environment. At inference time, click any point in the **Latent Space** panel to set the goal - the robot will navigate there autonomously.

### 6. Deploy

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



---

## Community

Share your trained weights, datasets, and demo videos on our [Discord](https://discord.com/channels/1487132795833684228/1487132796920004640). Contributed model weights feed into a federated **Model Soup** - enabling generalization across diverse home environments without sharing raw data.

## License

- **Software** - [GPLv3](LICENSE). Open to audit, modify, and contribute.
- **Your locally-trained weights** - 100% yours, no restrictions.
- **Pre-trained / aggregated weights** distributed by Embrained - governed by the Embrained Open-Weights EULA (non-commercial use permitted; commercial use requires a license). See [LICENSE](LICENSE) for full terms.
