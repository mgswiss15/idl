"""
utils.py
--------
Shared utilities: reproducibility, device selection, AverageMeter.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set all random seeds for full reproducibility.
    Call this before creating models or data loaders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slight performance trade-off)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Return the best available device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    return dev


class AverageMeter:
    """Tracks a running mean – used for loss/accuracy during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
