"""
train.py
--------
Train an MLP or CNN on CIFAR-10 and save the best checkpoint.

Usage examples
--------------
  python train.py --model cnn
  python train.py --model mlp --epochs 30 --lr 1e-3
  python train.py --model cnn --epochs 50 --batch_size 64 --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import get_dataloaders
from models import get_model, count_parameters
from utils import set_seed, get_device, AverageMeter


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_m  = AverageMeter()
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_m.update(loss.item(), images.size(0))
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return loss_m.avg, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_m  = AverageMeter()
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        loss_m.update(loss.item(), images.size(0))
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return loss_m.avg, correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = args.data_dir,
        model_type  = args.model,
        batch_size  = args.batch_size,
        val_fraction= 0.1,
        augment     = not args.no_augment,
        seed        = args.seed,
    )

    # Model
    model = get_model(args.model).to(device)
    print(f"\nModel : {args.model.upper()}")
    print(f"Params: {count_parameters(model):,}")
    print(f"Device: {device}\n")

    # Optimiser & scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    best_ckpt     = out_dir / f"best_{args.model}.pt"

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        lr  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  "
              f"{va_loss:>8.4f}  {va_acc:>7.2%}  {lr:>8.2e}  {elapsed:>5.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch":      epoch,
                "model":      args.model,
                "state_dict": model.state_dict(),
                "val_acc":    va_acc,
                "args":       vars(args),
            }, best_ckpt)

    print(f"\nBest val accuracy: {best_val_acc:.2%}")
    print(f"Checkpoint saved : {best_ckpt}")

    # Save training history
    hist_path = out_dir / f"history_{args.model}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved    : {hist_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MLP or CNN on CIFAR-10")
    p.add_argument("--model",        type=str,   default="cnn",
                   choices=["mlp", "cnn"], help="Model architecture")
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--data_dir",     type=str,   default="./data")
    p.add_argument("--out_dir",      type=str,   default="./checkpoints")
    p.add_argument("--no_augment",   action="store_true",
                   help="Disable data augmentation (CNN only)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
