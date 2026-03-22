"""
compare.py
----------
Load both trained checkpoints, evaluate on the test set, and produce a
side-by-side comparison report with plots.

Outputs (saved to --out_dir):
  comparison_curves.png   – training / validation loss & accuracy curves
  comparison_bar.png      – per-class accuracy bar chart
  comparison_report.txt   – plain-text summary table

Usage
-----
  python compare.py
  python compare.py --ckpt_dir ./checkpoints --out_dir ./results
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import get_dataloaders, CIFAR10_CLASSES
from models import get_model, count_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    for images, labels in loader:
        preds_all.append(model(images.to(device)).argmax(1).cpu())
        labels_all.append(labels)
    return torch.cat(labels_all).numpy(), torch.cat(preds_all).numpy()


def load_model_and_history(name: str, ckpt_dir: Path, device):
    ckpt_path = ckpt_dir / f"best_{name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {ckpt_path}\n"
            f"Run:  python train.py --model {name}"
        )
    state = torch.load(ckpt_path, map_location=device)
    model = get_model(name).to(device)
    model.load_state_dict(state["state_dict"])

    hist_path = ckpt_dir / f"history_{name}.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else None

    return model, state, history


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

COLORS = {"mlp": "#e76f51", "cnn": "#2a9d8f"}


def plot_curves(histories: dict, out_path: Path):
    """Plot loss and accuracy curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, hist in histories.items():
        if hist is None:
            continue
        epochs = range(1, len(hist["train_loss"]) + 1)
        c = COLORS[name]
        label = name.upper()
        axes[0].plot(epochs, hist["val_loss"],   color=c, label=f"{label} val",  lw=2)
        axes[0].plot(epochs, hist["train_loss"], color=c, label=f"{label} train", lw=1.5, linestyle="--", alpha=0.7)
        axes[1].plot(epochs, hist["val_acc"],    color=c, label=f"{label} val",  lw=2)
        axes[1].plot(epochs, hist["train_acc"],  color=c, label=f"{label} train", lw=1.5, linestyle="--", alpha=0.7)

    for ax, title, ylabel in zip(
        axes,
        ["Cross-Entropy Loss", "Accuracy"],
        ["Loss", "Accuracy"],
    ):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_class_bar(results: dict, out_path: Path):
    """Per-class accuracy bar chart."""
    x  = np.arange(len(CIFAR10_CLASSES))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (name, (y_true, y_pred)) in enumerate(results.items()):
        cm        = confusion_matrix(y_true, y_pred)
        per_class = cm.diagonal() / cm.sum(axis=1)
        offset    = (i - 0.5) * w
        bars      = ax.bar(x + offset, per_class, w, label=name.upper(),
                           color=COLORS[name], alpha=0.85)
        for bar, val in zip(bars, per_class):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy: MLP vs CNN", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    raw_data  = {}
    summaries = {}

    for name in ("mlp", "cnn"):
        print(f"\n{'='*50}")
        print(f"  Loading {name.upper()}")
        print(f"{'='*50}")
        model, state, history = load_model_and_history(name, ckpt_dir, device)

        # Get test loader matching the model type
        _, _, test_loader = get_dataloaders(
            data_dir   = args.data_dir,
            model_type = name,
            batch_size = args.batch_size,
            augment    = False,
            seed       = 42,
        )

        y_true, y_pred = predict(model, test_loader, device)
        acc = accuracy_score(y_true, y_pred)

        histories[name] = history
        raw_data[name]  = (y_true, y_pred)
        summaries[name] = {
            "test_acc":   acc,
            "val_acc":    state["val_acc"],
            "epochs":     state["epoch"],
            "params":     count_parameters(model),
        }

        print(f"  Test accuracy : {acc:.4f} ({acc:.2%})")
        print(f"  Val  accuracy : {state['val_acc']:.4f} ({state['val_acc']:.2%})")
        print(f"  Parameters    : {count_parameters(model):,}")
        print(f"\n  Per-class breakdown:")
        print(classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, digits=4))

    # --- Plots ---
    plot_curves(histories,   out_dir / "comparison_curves.png")
    plot_per_class_bar(raw_data, out_dir / "comparison_bar.png")

    # --- Text report ---
    report_lines = [
        "=" * 60,
        "  CIFAR-10 Model Comparison Report",
        "=" * 60,
        "",
        f"  {'Metric':<22} {'MLP':>12} {'CNN':>12}",
        f"  {'-'*46}",
    ]
    metrics = [
        ("Test Accuracy",  "test_acc",  ".2%"),
        ("Val  Accuracy",  "val_acc",   ".2%"),
        ("Parameters",     "params",    ",d"),
        ("Training Epochs","epochs",    "d"),
    ]
    for label, key, fmt in metrics:
        mlp_val = summaries["mlp"][key]
        cnn_val = summaries["cnn"][key]
        report_lines.append(
            f"  {label:<22} {format(mlp_val, fmt):>12} {format(cnn_val, fmt):>12}"
        )

    report_lines += ["", "=" * 60, ""]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    txt_path = out_dir / "comparison_report.txt"
    txt_path.write_text(report_text)
    print(f"Saved: {txt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare MLP and CNN on CIFAR-10")
    p.add_argument("--ckpt_dir",   type=str, default="./checkpoints")
    p.add_argument("--out_dir",    type=str, default="./results")
    p.add_argument("--data_dir",   type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    compare(parse_args())
