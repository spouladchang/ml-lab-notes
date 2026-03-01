# shallow-nn-playground

Exploring how small knobs — activation functions, weight initialization, feature scaling, and optimizers — change whether a neural network learns anything at all.

## The Setup

Binary classification on a synthetic blob dataset (800 samples, 2 features). Originally 4 classes, then remapped into 2 overlapping classes to add non-linearity. Two network architectures are compared throughout:

- **Shallow**: 1 hidden layer, 4 neurons → sigmoid output
- **Deep**: 4 hidden layers, 4 neurons each → sigmoid output

All models use SGD (or SGD variants), binary cross-entropy loss, and a 70/30 train/validation split.

---

## The 10 Experiments — Quick Summary

### Section 1: Activation Functions (Shallow Networks)

| Model | Activation | LR | Epochs | Batch | Init std | Result |
|-------|------------|-----|--------|-------|----------|--------|
| 1 | Sigmoid | 0.1 | 250 | 32 | 0.01 | Stuck ~50% |
| 2 | Sigmoid | 0.1 | 500 | 16 | 0.01 | Stuck ~50% |
| 3 | Sigmoid | 0.3 | 500 | 16 | 0.01 | Unstable, stuck ~50% |
| 4 | ReLU | 0.1 | 250 | 32 | 0.01 | **94.25% accuracy** ✅ |

Models 1–3 never learn — sigmoid with tiny random weights causes vanishing gradients from the start. Switching to ReLU (Model 4) breaks the deadlock immediately.

### Section 2: Shallow vs. Deep Networks & Weight Initialization

| Model | Architecture | Activation | LR | Epochs | Batch | Init | Result |
|-------|-------------|------------|----|--------|-------|------|--------|
| 5 | Deep (4 layers) | Sigmoid | 0.3 | 1000 | 16 | 0.01 | Stuck ~50% — worse than shallow |
| 6 | Deep (4 layers) | ReLU | 0.1 | 250 | 16 | 0.01 | Stuck ~50% — dead neurons |
| 7 | Deep (4 layers) | ReLU | 0.1 | 250 | 16 | He (√2/4) | **~94% val accuracy by epoch 2** ✅ |

Going deep makes the vanishing gradient problem worse (Model 5 — even 1000 epochs don't help). Deep + ReLU still fails with std=0.01 (Model 6 — neurons die before they can learn). The fix: **He initialization** sets std=√(2/N), giving ReLU neurons the right signal strength from epoch 1. Model 7 is the big "aha" of this notebook.

### Section 3: Scaled vs. Unscaled Features

| Model | Architecture | Activation | Features | Result |
|-------|-------------|------------|----------|--------|
| 4 | Shallow | ReLU | Scaled | 94.25% ✅ |
| 8 | Shallow | ReLU | Unscaled | Stuck ~50% ❌ |

Same model, same hyperparameters — only the feature scaling changes. Without StandardScaler, the optimizer can't navigate the loss landscape. Shows that scaling isn't optional.

### Section 4: Optimization Algorithms

| Model | Architecture | Activation | Optimizer | LR | Epochs | Batch | Result |
|-------|-------------|------------|-----------|-----|--------|-------|--------|
| 9 | Shallow | Sigmoid | SGD + Momentum (0.99) | 0.01 | 500 | 16 | Stuck ~50% |
| 10 | Shallow | Sigmoid | NAG (Nesterov, 0.99) | 0.01 | 500 | 16 | Stuck ~50% |

Better optimizers don't save you from bad activation choices. Momentum and Nesterov both fail here because the root problem — sigmoid with tiny weights — prevents any meaningful gradient flow. Optimizer choice matters, but only after the architecture fundamentals are right.

---

## Key Takeaways

1. **Sigmoid + tiny weight init = no learning.** The gradients vanish before reaching early layers, especially in deeper networks.
2. **ReLU alone isn't enough.** With std=0.01, most ReLU neurons output zero from the start and never recover (dead neurons).
3. **He initialization is the unlock.** std=√(2/N) gives ReLU networks the right starting scale. Model 7 reaches 94% validation accuracy by epoch 2.
4. **Feature scaling is non-negotiable.** Unscaled features broke an otherwise working model (Model 8 vs Model 4).
5. **Depth amplifies initialization problems.** Deep + sigmoid failed harder than shallow + sigmoid, even with 4× more epochs.

---

## Stack

- Python, TensorFlow / Keras, scikit-learn, NumPy, Matplotlib
- Dataset: `sklearn.datasets.make_blobs` (800 samples, 4 original classes → relabeled to 2)

## Status

Complete. 10 models trained and visualized. Decision boundaries plotted for best shallow (Model 4) and best deep model (Model 7).
