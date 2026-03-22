"""
evaluate.py
-----------
Load a saved checkpoint and evaluate it on the CIFAR-10 test set.
Prints per-class accuracy and saves a confusion matrix image.

Usage
-----
  python evaluate.py --model cnn
  python evaluate.py --model mlp --ckpt_dir ./checkpoints
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from data_loader import get_dataloaders, CIFAR10_CLASSES
from models import get_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm_norm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt     = ckpt_dir / f"best_{args.model}.pt"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Run:  python train.py --model {args.model}"
        )

    # Load checkpoint
    state = torch.load(ckpt, map_location=device)
    model = get_model(args.model).to(device)
    model.load_state_dict(state["state_dict"])
    print(f"Loaded {args.model.upper()} checkpoint (epoch {state['epoch']}, "
          f"val_acc={state['val_acc']:.2%})")

    # Data (no augmentation for evaluation)
    _, _, test_loader = get_dataloaders(
        data_dir   = args.data_dir,
        model_type = args.model,
        batch_size = args.batch_size,
        augment    = False,
        seed       = 42,
    )

    # Run
    y_true, y_pred = run_evaluation(model, test_loader, device)

    # Report
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc:.2%})\n")
    print(classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, digits=4))

    # Confusion matrix
    out_dir = ckpt_dir
    plot_confusion_matrix(y_true, y_pred, CIFAR10_CLASSES,
                          out_dir / f"confusion_{args.model}.png")

    return acc, y_true, y_pred


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint")
    p.add_argument("--model",      type=str, default="cnn",
                   choices=["mlp", "cnn"])
    p.add_argument("--ckpt_dir",   type=str, default="./checkpoints")
    p.add_argument("--data_dir",   type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
