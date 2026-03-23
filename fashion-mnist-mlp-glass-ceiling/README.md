# fashion-mnist-mlp-glass-ceiling

> **Part of [`ml-lab-notes`](https://github.com/spouladchang/ml-lab-notes/tree/main)** — a collection of hands-on ML experiments, each a focused deep dive into one concept, tested and visualized.

A rigorous **6-phase controlled ablation study** that extracts the maximum possible accuracy
from a pure Dense (MLP) network on Fashion-MNIST — one variable at a time — until there
is nothing left to tune. The endpoint is not a number; it is a per-class F1 table that
shows exactly which errors are tunable and which are architectural.

---

## Results

| Phase | Variable introduced | Test accuracy | Δ vs baseline |
|-------|--------------------|--------------:|:-------------:|
| **P0** — True baseline | Fixed 2×128 ReLU, Adam, nothing tuned | 87.82% | — |
| **P1** — Architecture search | Layers 1–4, neurons 64–512, relu/tanh/gelu | 88.68% | +0.86% |
| **P2** — Dropout | Dropout rate 0.0–0.5 per layer | 88.83% | +0.15% |
| **P3** — Batch Normalization | Per-layer BN flag (on/off) | 89.17% | +0.34% |
| **P4** — AdamW + weight decay ★ | AdamW + weight_decay log-searched | **89.28%** | **+0.11%** |
| **P5** — Hyperband + LR | Hyperband algorithm + LR log-searched | 89.25% | −0.03% |

**Total gain from six phases of exhaustive tuning: +1.46%**

The diminishing deltas tell the story: each new variable contributed less than the previous one,
and the final phase found nothing. The architecture is saturated.

---

## Champion architecture (P4)

```
Input → Flatten (784)
Dense(384, activation=gelu)
Dropout(0.2)
Dense(10, softmax)

Optimizer: AdamW(lr=0.001, weight_decay=0.003)
```

One wide GELU layer with light dropout. Every phase from P1 onward converged to a single-layer
architecture — the tuner consistently rejected depth without stronger regularization support.

---

## The glass ceiling — per-class F1

| Class | P0 | P4 (best) | Verdict |
|-------|:--:|:---------:|---------|
| T-shirt/top | 82.2 | 84.8 | Tunable |
| Trouser | 97.8 | 98.2 | Near ceiling |
| **Pullover** | **79.0** | **81.0** | Structural limit |
| Dress | 88.6 | 90.2 | Tunable |
| Coat | 79.5 | 81.8 | Marginal |
| Sandal | 96.4 | 96.6 | Near ceiling |
| **Shirt** | **69.6** | **72.5** | **Structural limit** |
| Sneaker | 94.1 | 94.7 | Near ceiling |
| Bag | 96.9 | 97.0 | Near ceiling |
| Ankle boot | 95.2 | 95.9 | Near ceiling |

Shirt reached only 72.5% F1 after six phases of tuning. Pullover barely cleared 80%.
These are not tuning failures — they are garments that differ primarily in spatial cut and
drape. A flattened 784-pixel input has no way to represent "the collar is here, the hem
is there". No regularization technique changes that.

CNNs reach **95–96%** on this dataset because they build exactly those spatial features.
The ~5–6% gap between this MLP ceiling and a baseline CNN is architectural, not a tuning gap.
The per-class table shows which categories it lives in.

---

## The experiment design

The central rule: **one new variable per phase**. Every delta is therefore a causal statement.

**Why GELU in Phase 1?** It is an architecture choice, not a regularization choice. No reason to withhold it.

**Why neurons up to 512?** The 256-neuron ceiling in earlier studies was arbitrary. Fashion-MNIST has 784 input features — the tuner should decide the right width.

**Why BN before the optimizer swap?** BN affects gradient scale. If it changes what the optimal optimizer setting would be, that effect should be isolated in P3, not hidden inside a simultaneous change with P4.

**Why search weight_decay?** Hardcoding `wd=0.004` (common in tutorials) confounds "does AdamW help?" with "is 0.004 the right value?". P4 answers only the first question by letting the tuner pick the decay.

**Why LR last?** It is the final fixed hyperparameter. Isolating it to P5 keeps every prior delta attributable to only its stated variable.

---

## Notable findings

**BN was not selected.** The P3 champion chose `BN=no`. The tuner explored BatchNorm and rejected it for this architecture. The +0.34% gain in P3 came from higher dropout (0.1→0.2), not BN. BN appeared in only one layer of P5's 3-layer architecture — suggesting it is useful in specific configurations, not as a default.

**GELU beat ReLU when weight decay was available.** The P4 champion switched from ReLU to GELU only after AdamW's weight decay was added. This makes sense: GELU's smoother gradient landscape benefits from decoupled regularization.

**P5 used a higher LR schedule that partially conflicts with the LR search.** The `exp(-0.1)` scheduler runs in every phase. In P5, Hyperband found `lr=0.00039` — but the scheduler also decayed it throughout training, so the effective LR was much lower than the searched value by later epochs. The result is still valid, but the LR dimension in P5 is slightly confounded with the scheduler.

---

## How to run

Each phase is **self-contained** — every section re-imports, reloads data, and redefines all helpers. Jump to any phase and run it cold.

**Full run (recommended):**
```bash
# Google Colab (GPU runtime) or local Jupyter
jupyter notebook fashion_mnist_mlp_glass_ceiling.ipynb
```

**Phase 5 warm start:** Phase 5 reads `p4_defaults` from Phase 4. If running in isolation, the fallback config at the top of the cell is used — replace it with your actual P4 champion values.

**Requirements:**
```
tensorflow >= 2.12  |  keras-tuner >= 1.3  |  scikit-learn  |  numpy  |  matplotlib  |  seaborn
```
Installed automatically via `!pip install keras-tuner -q`.

---

## Output files

| File | Use |
|------|-----|
| `fashion_mnist_mlp_champion.keras` | Full model — easiest for transfer learning |
| `fashion_mnist_mlp_champion.weights.h5` | Weights only — portable |

**Transfer learning:**
```python
base  = tf.keras.models.load_model("fashion_mnist_mlp_champion.keras")
x     = base.layers[-2].output
out   = keras.layers.Dense(N_CLASSES, activation="softmax")(x)
model = keras.Model(base.input, out)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

---

## Stack

Python 3.10 · TensorFlow/Keras · Keras Tuner (Bayesian Optimization, Hyperband) · scikit-learn · NumPy · Matplotlib · Seaborn

Dataset: `keras.datasets.fashion_mnist` — 60 000 train / 10 000 test · 28×28 greyscale · 10 classes

---

## Related

**[`nn_knobs_exploration / shallow-nn-playground`](../nn_knobs_exploration/)** — how activation functions, weight initialization, feature scaling, and optimizers affect whether a network learns at all. 10 models, synthetic blob dataset, full decision boundary plots.

---

## Status

Complete. Six phases. Champion: P4 (AdamW + searched weight decay) at **89.28%**.  
Glass ceiling confirmed: Shirt (72.5% F1 max) and Pullover (81.0% F1 max) do not respond to MLP tuning.
