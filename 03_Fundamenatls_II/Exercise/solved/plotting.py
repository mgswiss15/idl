"""THWS/MAI/Introduction to Deep Learning - Assignment 2 - plotting

Magda Gregorová, April 2026

These functions are provided. You do not need to modify this file.
"""

import torch
import matplotlib.pyplot as plt


def plot_data(X, y):
    """Plot size vs price and distance vs price side by side.

    Args:
        X: torch.tensor of shape (N, 4) - input features (age, size, distance, floor)
        y: torch.tensor of shape (N, 1) - target prices
    """
    x_size     = X[:, 1].numpy()
    x_distance = X[:, 2].numpy()
    y_vals     = y[:, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(x_size, y_vals, alpha=0.4, s=15, color='steelblue')
    axes[0].set_xlabel('size (standardised)')
    axes[0].set_ylabel('price (standardised)')
    axes[0].set_title('Size vs Price')

    axes[1].scatter(x_distance, y_vals, alpha=0.4, s=15, color='steelblue')
    axes[1].set_xlabel('distance (standardised)')
    axes[1].set_ylabel('price (standardised)')
    axes[1].set_title('Distance vs Price')

    fig.tight_layout()
    plt.show()


def plot_fit(X, y, model):
    """Plot size vs price with the model's predictions overlaid.

    Varies size across its observed range while fixing other features at their
    mean (zero, since data is standardised).

    Args:
        X:     torch.tensor of shape (N, 4) - input features
        y:     torch.tensor of shape (N, 1) - target prices
        model: object with a forward(X) method returning shape (N, 1)
    """
    x_size = X[:, 1].numpy()
    y_vals = y[:, 0].numpy()

    # build a grid varying size, other features fixed at 0 (their mean)
    size_grid = torch.linspace(X[:, 1].min(), X[:, 1].max(), 200)
    X_grid = torch.zeros(200, 4)
    X_grid[:, 1] = size_grid

    with torch.no_grad():
        y_grid = model.forward(X_grid).squeeze().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_size, y_vals, alpha=0.4, s=15, color='steelblue', label='data')
    ax.plot(size_grid.numpy(), y_grid, color='crimson', linewidth=2, label='model fit')
    ax.set_xlabel('size (standardised)')
    ax.set_ylabel('price (standardised)')
    ax.set_title('Linear model fit — size vs price')
    ax.legend()
    fig.tight_layout()
    plt.show()
