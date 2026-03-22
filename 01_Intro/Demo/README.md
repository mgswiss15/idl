# CIFAR-10 Classification: MLP vs CNN

A clean, reproducible PyTorch project that trains and compares two architectures on CIFAR-10:
an **MLP** (fully-connected baseline) and a **CNN** (VGG-style convolutional network).

---

## Project Structure

```
cifar10/
├── data_loader.py   # CIFAR-10 download, transforms, and DataLoaders
├── models.py        # MLP and CNN architecture definitions
├── train.py         # Training loop with checkpointing
├── evaluate.py      # Test-set evaluation and confusion matrix
├── compare.py       # Side-by-side model comparison with plots
├── utils.py         # Seed setting, device selection, AverageMeter
├── requirements.txt # Python dependencies
└── README.md        # This file
```

After running the scripts, two additional directories appear:

```
checkpoints/         # Saved model weights and training history
results/             # Comparison plots and text report
data/                # CIFAR-10 dataset (auto-downloaded)
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA-enabled PyTorch wheel from https://pytorch.org
> before running `pip install -r requirements.txt`.

---

## Quickstart (recommended order)

```bash
# 1. Train the CNN  (~15–40 min on CPU, ~3–8 min on GPU)
python train.py --model cnn

# 2. Train the MLP  (~5–15 min on CPU, ~1–3 min on GPU)
python train.py --model mlp

# 3. Evaluate each model on the test set
python evaluate.py --model cnn
python evaluate.py --model mlp

# 4. Generate the comparison report and plots
python compare.py
```

---

## Script Reference

### `train.py` — Train a model

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `cnn` | Architecture: `mlp` or `cnn` |
| `--epochs` | `40` | Number of training epochs |
| `--batch_size` | `128` | Mini-batch size |
| `--lr` | `3e-4` | Initial learning rate (AdamW) |
| `--weight_decay` | `1e-4` | L2 regularisation coefficient |
| `--seed` | `42` | Random seed |
| `--data_dir` | `./data` | Dataset cache directory |
| `--out_dir` | `./checkpoints` | Where to save checkpoints |
| `--no_augment` | off | Disable data augmentation (CNN) |

**Outputs saved to `checkpoints/`:**
- `best_cnn.pt` / `best_mlp.pt` — Best checkpoint (by validation accuracy)
- `history_cnn.json` / `history_mlp.json` — Loss and accuracy per epoch

---

### `evaluate.py` — Evaluate on the test set

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `cnn` | Which checkpoint to load |
| `--ckpt_dir` | `./checkpoints` | Where checkpoints live |
| `--data_dir` | `./data` | Dataset directory |
| `--batch_size` | `256` | Evaluation batch size |

**Outputs saved to `checkpoints/`:**
- `confusion_cnn.png` / `confusion_mlp.png` — Normalised confusion matrix

Prints per-class precision, recall, F1-score, and overall accuracy.

---

### `compare.py` — Side-by-side comparison

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt_dir` | `./checkpoints` | Checkpoint directory |
| `--out_dir` | `./results` | Where to save comparison outputs |
| `--data_dir` | `./data` | Dataset directory |
| `--batch_size` | `256` | Evaluation batch size |

**Outputs saved to `results/`:**
- `comparison_curves.png` — Training and validation loss/accuracy curves
- `comparison_bar.png` — Per-class accuracy bar chart
- `comparison_report.txt` — Plain-text summary table

---

## Reproducibility

All randomness is controlled through a single `--seed` flag (default `42`), which seeds:

- Python's `random` module
- NumPy
- PyTorch (CPU and all CUDA devices)
- The train/validation split via a seeded `torch.Generator`
- cuDNN deterministic mode

Running the same command twice on the same hardware will produce identical results.

> **Note:** Exact numerical reproducibility across different GPUs, drivers, or PyTorch
> versions is not guaranteed due to floating-point non-determinism in some CUDA kernels.
> On the same machine and environment, results will match exactly.

---

## Expected Results

These figures are approximate and may vary slightly by hardware:

| Metric | MLP | CNN |
|--------|-----|-----|
| Test Accuracy | ~55–58 % | ~87–90 % |
| Parameters | ~3.5 M | ~5.8 M |

The CNN's spatial inductive bias gives it a substantial advantage over the flat MLP.

---

## Architecture Details

### MLP
- Input: 3 072-dimensional flattened image vector
- Three hidden layers: 1 024 → 512 → 256 units
- Each layer: Linear → BatchNorm1d → ReLU → Dropout (0.3)
- Output: 10-class logits

### CNN
- Three convolutional stages, each with two Conv→BN→ReLU blocks and MaxPool
- Channels: 64 → 128 → 256
- Classifier head: 4 096 → 512 → 256 → 10 with BatchNorm and Dropout (0.5)
- Training uses random horizontal flip and random crop augmentation

### Shared training setup
- Optimiser: AdamW
- Scheduler: Cosine annealing (η_min = 1e-6)
- Loss: Cross-entropy with label smoothing (ε = 0.1)
- Validation fraction: 10 % of training data
