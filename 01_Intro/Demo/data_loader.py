"""
data_loader.py
--------------
CIFAR-10 data loading and preprocessing for MLP and CNN models.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Normalization stats computed on CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(model_type: str = "cnn", augment: bool = True):
    """
    Return train and test transforms for the given model type.

    Args:
        model_type: "mlp" or "cnn"
        augment:    Apply data augmentation to training set (CNN only)
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    if model_type == "mlp":
        # MLP expects a flat vector – no augmentation to keep it simple
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(lambda x: x.view(-1)),   # 3*32*32 = 3072
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    else:
        # CNN keeps spatial structure; optionally augments training data
        aug_steps = (
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
            if augment else []
        )
        train_tf = transforms.Compose(aug_steps + [transforms.ToTensor(), normalize])
        test_tf  = transforms.Compose([transforms.ToTensor(), normalize])

    return train_tf, test_tf


def get_dataloaders(
    data_dir: str = "./data",
    model_type: str = "cnn",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    augment: bool = True,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Download (if needed) and return DataLoaders for train, val, and test splits.

    Args:
        data_dir:     Directory where CIFAR-10 will be cached.
        model_type:   "mlp" or "cnn" – controls transforms.
        batch_size:   Mini-batch size.
        val_fraction: Fraction of training data held out for validation.
        augment:      Whether to use data augmentation (CNN train split only).
        num_workers:  DataLoader worker processes.
        seed:         Random seed for reproducible train/val split.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_tf, test_tf = get_transforms(model_type, augment)

    # Full training set (50 000 images) and test set (10 000 images)
    full_train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds    = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    # Reproducible train / val split
    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[data_loader] model_type={model_type} | "
          f"train={n_train}  val={n_val}  test={len(test_ds)} | "
          f"batch_size={batch_size}")
    return train_loader, val_loader, test_loader
