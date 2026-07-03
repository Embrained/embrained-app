# Plexus Visual Navigation — Model Development Guide

## Architecture Overview

Your navigation pipeline has two stages:

```
Camera Frame → [CVE Encoder] → 32-dim latent → [CQL Policy] → Action
                                                    ↑
                                            [GoalClassifier] → Reward signal
```

1. **CVE (Contrastive Visuomotor Encoder)** — Pre-trained, frozen. Maps 64×64 images to 32-dim latents.
2. **GoalClassifier** — Binary CNN. Outputs P(goal) ∈ [0,1] for reward shaping.
3. **CQL (Conservative Q-Learning)** — Offline RL policy. Takes 32-dim latent, outputs action.

---

## Model Progression Roadmap

### Model 1: Sofa Proximity Classifier ✅
Binary CNN that distinguishes "close to sofa" vs "exploring".

**Train:**
```bash
# From the embrained-app directory:
python seek/train_classifier.py sofa
```

**What it does:**
- Loads positive examples from `data/sofa/` (your curated 100 images)
- Samples negatives from `data/markov_*/images/`
- Trains with augmentation + weighted sampling
- Saves to `data/sofa_classifier.pth`
- Generates prediction grid and training curves in `images/`

**Adding more examples:** Just add more `.jpg` files to `data/sofa/` and retrain.

**Key metric:** F1 > 0.90 on validation set.

---

### Model 2: Sofa-Seeking CQL
Offline RL policy trained with classifier-shaped rewards.

**Prerequisites:** Trained `sofa_classifier.pth` in `data/`

**Train:**
```bash
# From the embrained-app directory:
python seek/train_seek_cql.py sofa
```

**What it does:**
- Loads the sofa classifier for reward computation
- For each training transition, computes: `reward = (P(sofa) - 0.5) * 2.0`
- Terminal states (P(sofa) > 0.85) get reward=3.0 and action=INTENTIONAL_STOP
- Saves CQL model to `data/cve_32d_*-sofa_seek_cql_model.pth`

**Deploy:** Load the model in the Plexus UI via the model selector. It will automatically:
- Use 32-dim input (no goal concatenation)
- Navigate without needing explicit goals set in the UI

---

### Model 3: TV-Seeking (Template)
Same architecture, different target.

```bash
# 1. Create data/tv/ and add ~100 TV close-up images
# 2. Train classifier
python seek/train_classifier.py tv
# 3. Train CQL
python seek/train_seek_cql.py tv
```

---

### Model 4: Multi-Goal Policy (Future)
A single CQL policy conditioned on goal type. Requires concatenating a goal embedding (e.g., one-hot or learned) to the 32-dim CVE latent:
- Input: `[z_current (32d) | goal_id (8d)]` → 40-dim
- Needs architectural changes to `CQLNetwork(input_dim=40, ...)`

---

### Model 5: Spatial CVE (Future)
Enhanced CVE that encodes spatial awareness (distance to obstacles, room region). Would require data augmentation with position labels from overhead camera or SLAM.

---

## Evaluation

### Manual Displacement Trials

1. Load a trained `*_seek_cql` model in Plexus
2. Place robot at a random location
3. Let it navigate until STOP or timeout
4. Pick it up, displace to a new location
5. Repeat 10-20 times per session

### Automated Scoring

```bash
# From the embrained-app directory:
python seek/evaluate_navigation_trials.py markov_SESSION_NAME --goal sofa
```

**Metrics:**
- **Success Rate** — % of trials where robot stops at P(sofa) > 0.85
- **Steps to Target** — Mean/median steps for successful trials
- **Max Confidence** — Peak P(sofa) during each trial

**Target:** Success rate > 60% with mean steps < 30

---

## Data Collection Best Practices

### For Classifier Training
- **Positive examples:** 100+ curated images in `data/<target>/`
- **Diversity matters:** Different angles, lighting, distances from target
- **Edge cases:** Include borderline examples (partial sofa view, very close texture)

### For CQL Training  
- **Random exploration data:** The robot's existing `markov_*` sessions
- **More is better:** 10+ sessions with 1000+ frames each
- **Action diversity:** Ensure forward, left, right, reverse are all well-represented

### Recording New Data
Just run the robot in random walk mode. The app saves sessions to `data/markov_TIMESTAMP/` automatically.

---

## File Reference

| File | Purpose |
|------|---------|
| `modules/goal_classifier.py` | GoalClassifier model definition |
| `seek/train_classifier.py` | Classifier training script |
| `seek/train_seek_cql.py` | CQL training orchestrator |
| `seek/evaluate_navigation_trials.py` | Trial evaluation script |
| `backend/train_cql.py` | Core CQL training (modified for `*_seek`) |
| `modules/planner.py` | Inference-time action selection |
| `scripts/plot_sofa_latent.py` | PCA latent space visualization |

---

## Troubleshooting

### Classifier F1 is low
- Add more positive examples to `data/sofa/`
- Check that positives are genuinely "close-up sofa" views, not distant glimpses
- Try `model_size='enormous'` for more capacity

### CQL doesn't learn to navigate
- Check classifier terminal rate: should be 5-15% of transitions
- Try lower `alpha` (0.01) for less conservative Q-learning
- Increase `num_epochs` to 400-500
- Verify action distribution in training data

### Robot spins in circles
- Action imbalance in data — ensure forward actions are well-represented
- Check `valid_actions` mask in checkpoint

### Model loads but doesn't move
- Verify `_seek_cql` is in the model filename (planner detection pattern)
- Check console logs for "Successfully loaded" messages

---

## Quick Commands

```bash
# Train classifier
python seek/train_classifier.py sofa

# Train CQL
python seek/train_seek_cql.py sofa

# Evaluate trials
python seek/evaluate_navigation_trials.py markov_SESSION --goal sofa

# Visualize latent space (from cognitive-engine/)
python scripts/plot_sofa_latent.py

# Check for new goal categories
# Just create data/<goal_name>/ → add images → train classifier → train CQL
```
