"""
models.py
---------
Model definitions for CIFAR-10 classification.

  MLP  – fully-connected baseline (input: flattened 3072-d vector)
  CNN  – convolutional network with BatchNorm and Dropout
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Three-hidden-layer MLP for CIFAR-10.

    Architecture:
        3072 → 1024 → 512 → 256 → 10
        Each hidden layer: Linear → BatchNorm1d → ReLU → Dropout
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # --- block 1 ---
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # --- block 2 ---
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # --- block 3 ---
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # --- output ---
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU helper."""

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN(nn.Module):
    """
    Small VGG-style CNN for CIFAR-10.

    Architecture (input: 3×32×32):
        Conv stage 1: [64]×2  → MaxPool  → 64×16×16
        Conv stage 2: [128]×2 → MaxPool  → 128×8×8
        Conv stage 3: [256]×2 → MaxPool  → 256×4×4
        Classifier:   256*4*4 → 512 → 256 → 10
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # --- stage 1 ---
            ConvBlock(3,   64,  kernel_size=3, padding=1),
            ConvBlock(64,  64,  kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),                              # 64×16×16
            nn.Dropout2d(0.1),
            # --- stage 2 ---
            ConvBlock(64,  128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),                              # 128×8×8
            nn.Dropout2d(0.2),
            # --- stage 3 ---
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),                              # 256×4×4
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Return a model by name.

    Args:
        name:   "mlp" or "cnn"
        kwargs: passed to the model constructor (e.g. dropout=0.4)
    """
    name = name.lower()
    if name == "mlp":
        return MLP(**kwargs)
    elif name == "cnn":
        return CNN(**kwargs)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'mlp' or 'cnn'.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ("mlp", "cnn"):
        m = get_model(name)
        params = count_parameters(m)
        dummy = torch.zeros(4, 3072) if name == "mlp" else torch.zeros(4, 3, 32, 32)
        out = m(dummy)
        print(f"{name.upper():>3} | params={params:,} | output shape={tuple(out.shape)}")
