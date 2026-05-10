"""THWS/MAI/Introduction to Deep Learning - Session 6 - plotting

Magda Gregorová, May 2026

These functions are provided. You do not need to modify this file.
"""

import matplotlib.pyplot as plt

# THWS color palette
_PETROL  = '#005564'   # thwspetrol - train loss
_ORANGE  = '#FF6A00'   # thwsorange - val loss
_BLUE    = '#163C69'   # thwsblue
_GREY    = '#D9D9D9'   # thwsgrey

# line colors for multi-model comparisons
_COLORS = [
    ('#005564', '#FF6A00'),   # petrol / orange  (model 1)
    ('#163C69', '#B31B1B'),   # blue   / carnelian (model 2)
    ('#4A90A4', '#F5A623'),   # light petrol / amber (model 3)
]


def plot_losses(train_losses, val_losses, title='Training curve'):
    """Plot train and validation loss curves.

    Args:
        train_losses: list of float - training loss per epoch
        val_losses:   list of float - validation loss per epoch
        title:        str - plot title
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, train_losses, color=_PETROL, linewidth=2,
            label='train loss')
    ax.plot(epochs, val_losses,   color=_ORANGE, linewidth=2,
            linestyle='--', label='val loss')

    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title(title, color=_BLUE, fontweight='bold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_losses_compare(models, title='Model comparison'):
    """Plot train and validation loss curves for multiple models.

    Args:
        models: list of tuples (train_losses, val_losses, label)
                - train_losses: list of float
                - val_losses:   list of float
                - label:        str - model name for legend
        title:  str - plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, (train_losses, val_losses, label) in enumerate(models):
        c_train, c_val = _COLORS[i % len(_COLORS)]
        epochs = range(1, len(train_losses) + 1)

        axes[0].plot(epochs, train_losses, color=c_train, linewidth=2, label=label)
        axes[1].plot(epochs, val_losses,   color=c_val,   linewidth=2, label=label)

    for ax, split in zip(axes, ['train loss', 'val loss']):
        ax.set_xlabel('epoch')
        ax.set_ylabel('MSE loss')
        ax.set_title(split, color=_BLUE, fontweight='bold')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(title, color=_BLUE, fontweight='bold', fontsize=13)
    fig.tight_layout()
    plt.show()
