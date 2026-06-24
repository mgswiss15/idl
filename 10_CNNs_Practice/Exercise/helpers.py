"""
MAI/IDL SS26 - Pretraining demo. 

MG 24/6/2026
"""

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


def plot_data_examples(dataset):
    fig, axes = plt.subplots(1, 5, figsize=(10, 4))
    for ax, idx in zip(axes.ravel(), range(10)):
        img, label = dataset[idx]
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()


def get_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    mean, std, n_samples = 0., 0., 0.
    for images, _ in loader:
        batch_size = images.size(0)
        mean += images.mean(dim=[0, 2, 3]) * batch_size
        std += images.std(dim=[0, 2, 3]) * batch_size
        n_samples += batch_size
    mean /= n_samples
    std /= n_samples
    return mean, std

