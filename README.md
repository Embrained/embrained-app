# Embrained

Embrained, LLC is a neurorobotics company dedicated to democratizing Embodied AI[cite: 76]. We develop software and hardware platforms that interface with low-cost robots to collect large, high-quality datasets in unstructured home environments[cite: 31, 40]. We design architectures that allow standard consumer PCs to train and control complex neural navigation agents locally, securely, and privately[cite: 8, 24].

## Overview

This repository contains the source code for the **Embrained App** running the **Plexus** platform [cite: 21, 47], including the core Python backend cognitive engine, Live UI React dashboard, local PyTorch training orchestration, and bare-metal ESP32-S3 firmware modules[cite: 26, 31]. Our focus is on bringing advanced neurobiological and AI concepts—such as the Free Energy Principle, Latent Space navigation (via VQ-VAE), and Conservative Q-Learning (CQL) forward world models—out of purely theoretical bounds into physical, domestic environments[cite: 33, 34, 35].

We overcome the primary bottleneck in modern robotics—*data collection*—by making physical embodiment and training infrastructure highly accessible[cite: 40, 61]. By combining a standardized hardware layout with a local, decentralized AI stack, our "disaggregated intelligence" architecture offloads all heavy neural compute over WiFi to the user's local PC GPU, bypassing the need for expensive onboard computing[cite: 9, 30].

Breaking the robotics data barrier requires a trusted, privacy-native software stack that allows you to[cite: 27, 28]:
- Collect unstructured environmental transitions directly in the real world[cite: 47].
- Train custom VQ-VAE encoders to compress raw vision frames into discrete latent representations[cite: 34, 47].
- Map localized, user-labeled semantic regions (e.g., "kitchen", "front door") using local Vision-Language Models (VLMs)[cite: 36, 37].
- Deploy goal-conditioned, offline reinforcement learning policies running completely offline on your own hardware[cite: 35, 37].

## Quick Start

Welcome to the Embrained Beta Pilot! This guide will take you from an assembled box to your first successful autonomous navigation event in minutes[cite: 21, 47].

### 1. Repository Access & Installation

Clone the public GitHub repository and navigate into the project directory:

```bash
git clone https://github.com/Embrained/embrained-app.git
cd embrained-app

```

Run the automated setup script to initialize a local, isolated virtual environment. This guarantees the correct dependencies (Python 3.12, PyTorch, CUDA, Node.js) are compiled cleanly without conflicting with your system space.

**Windows:**

```bat
setup.bat

```

**Mac/Linux:**

```bash
./setup.sh

```

### 2. UI Initialization

Launch the local server to boot up the backend cognitive engine and serve the React dashboard user interface.

**Windows:**

```bat
start.bat

```

**Mac/Linux:**

```bash
./start.sh

```

Open your default web browser and navigate to `http://localhost:8080`. You are now ready to use the Embrained App for data collection, live telemetry viewing, and autonomy execution.

### 3. The CLI Training Pipeline

Once you have gathered sufficient raw transition data using teleoperation (saved automatically to the `/data` folder), use the command-line interface to train your model architecture. Ensure you have activated your virtual environment before executing these commands:

**Activate Virtual Environment:**

* *Windows:* `venv\Scripts\activate.bat`
* *Mac/Linux:* `source venv/bin/activate`

Run the following core terminal scripts in sequence from the project root directory:

1. **Pool Raw Data:**
Consolidates individual raw environmental transition recordings into a unified, clean training dataset.

```bash
python cognitive-engine/scripts/prepare_dataset.py

```

2. **Train the Encoder (Perception Manifold):**
Trains the VQ-VAE network to compress high-dimensional camera frames into discrete latent manifolds.



```bash
# Standard Turn-Based Architecture:
python cognitive-engine/backend/training/train_vqvae.py

# Or for our continuous action-chunking architecture (Q3 Milestone):
python cognitive-engine/backend/training/train_continuous_vae.py

```

3. **Train the Policy (Navigation Control):**
Trains the offline Conservative Q-Learning (CQL) controller to map perception latents into robust navigation motor commands.



```bash
python cognitive-engine/backend/training/train_discrete_fixed_goal_cql.py

```

### 4. Model Deployment

After the training pipeline completes, take your trained adapter weights and deploy them for live local execution.

1. Power on your Plexus robot and verify its local WiFi connection status via the app interface.


2. In the Embrained App UI (`http://localhost:8080`), navigate to the **Autonomy Panel**.


3. Select your newly generated models and VQ-VAE boundaries from the dropdowns.


4. Click **Start Autonomy**.



The Cognitive Engine will load your parameters locally and stream real-time topological action sequences to the robot, achieving secure, offline fixed-goal navigation right in your room.

### 5. Community Sharing & The "Model Soup"

Have you successfully trained a robust topological navigator or refined an edge-case visual boundary? We want to see it! We highly encourage users to share their successful weights, datasets, and execution videos on our [Discord Server](https://discord.com/channels/1487132795833684228/1487132796920004640).

By contributing anonymous model weights to the ecosystem, you participate in our **Federated Co-Training network**. This enables the compilation of a collective master "Model Soup," unlocking few-shot generalization across hundreds of unique domestic environments without ever compromising user privacy or transmitting raw video feeds.

---

## License & Terms of Use

Embrained operates under a strict dual-licensing legal framework designed to maximize open developer collaboration while rigidly protecting the network's data moat, security, and shared equity value.

### 1. Software Codebase (GNU GPLv3)

The Python software suite, local training orchestration engines, frontend React interfaces, and ESP32 firmware are open-sourced under the **GNU General Public License v3.0 (GPLv3)**. This copyleft license guarantees that the core application software remains explicitly open to audit, modification, and community contribution, protecting developers and ensuring local privacy compliance.

### 2. Neural Model Weights (Embrained Open-Weights License / EULA)

**Your Data, Your Models:** Any neural model weights you train locally on your own private data using this software suite are **100% your intellectual property** and are not subject to any Embrained restrictions.

**Pre-trained & Aggregated Models:** All pre-trained neural network mathematical constants, floating-point parameter matrices (`.safetensors` / `.bin` files), VQ-VAE codebook boundaries, and federated policy checkpoints distributed by Embrained or aggregated into the global "Model Soup" are **explicitly excluded** from the GPLv3 software grant. These specific pre-trained files are governed strictly by the *Embrained Open-Weights End User License Agreement (EULA)*.

* **Permitted Use:** Individual users are granted a non-exclusive, revocable, royalty-free license to utilize, evaluate, and fine-tune model weights locally for non-commercial, personal, or academic research objectives.
* **Commercial Restrictions:** Commercial redistribution, enterprise hosting, cloud deployment, or utilizing Embrained weights to bootstrap, distill, or train competing robotic navigation architectures is strictly prohibited without executing an explicit corporate licensing contract with Embrained, LLC.
* 
**Hardware Co-Dependency Clause:** Because our visual manifolds and navigation policies are mathematically bound to the exact physical embodiment parameters of the **Plexus robot** (including precise camera center height, chassis lens distortion, motor driver latency, and 2WD differential drive kinematics), weights are structurally optimized for unified hardware. The injection of uncalibrated or hardware-agnostic weight structures that degrade the shared federated layer is a violation of the network's terms of use.



For full legal terms, please review the included `LICENSE` file and our centralized policy resources.

