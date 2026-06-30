"""
MAI/IDL SS26 - VAE demo.

MG 30/6/2026
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def plot_data_examples(dataset, n=5):
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
    for ax, idx in zip(axes.ravel(), range(n)):
        img, label = dataset[idx]
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()


def plot_losses(train_losses):
    plt.plot(train_losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


def plot_vae_losses(recon_losses, kl_losses):
    plt.plot(recon_losses, label="Reconstruction")
    plt.plot(kl_losses, label="KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


def plot_examples_with_codes(model, dataset, device, n=5):
    """Show a few images next to their learned 2D latent code (AE)."""
    model.eval()
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.7))
    with torch.no_grad():
        for ax, idx in zip(axes.ravel(), range(n)):
            img, label = dataset[idx]
            z = model.encode(img.unsqueeze(0).to(device))
            z = z.squeeze(0).cpu()
            ax.imshow(img.squeeze(0), cmap="gray")
            ax.set_title(f"z=({z[0]:.2f}, {z[1]:.2f})", fontsize=9)
            ax.axis("off")
    plt.tight_layout()


def plot_latent_scatter_ae(model, dataset, device, n_points=2000):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    zs, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            z = model.encode(images)
            zs.append(z.cpu())
            labels.append(lbls)
            if sum(t.size(0) for t in zs) >= n_points:
                break
    zs = torch.cat(zs)[:n_points]
    labels = torch.cat(labels)[:n_points]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap="tab10", s=8)
    plt.colorbar(scatter, ticks=range(10), label="digit")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")


def plot_latent_scatter_vae(model, dataset, device, n_points=2000):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    mus, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            mu, logvar = model.encode(images)
            mus.append(mu.cpu())
            labels.append(lbls)
            if sum(t.size(0) for t in mus) >= n_points:
                break
    mus = torch.cat(mus)[:n_points]
    labels = torch.cat(labels)[:n_points]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="tab10", s=8)
    plt.colorbar(scatter, ticks=range(10), label="digit")
    plt.xlabel("mu[0]")
    plt.ylabel("mu[1]")


def plot_samples_grid(model, device, n=64, latent_dim=2):
    """Sample z ~ N(0,I) and decode. Works for AE or VAE (both have .decode())."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        x_hat = model.decode(z).cpu()

    rows = cols = int(n ** 0.5)
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for ax, img in zip(axes.ravel(), x_hat):
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.axis("off")
    plt.tight_layout()


def plot_interpolation(model, dataset, device, idx1, idx2, n_steps=10, is_vae=False):
    model.eval()
    img1, _ = dataset[idx1]
    img2, _ = dataset[idx2]

    with torch.no_grad():
        if is_vae:
            mu1, _ = model.encode(img1.unsqueeze(0).to(device))
            mu2, _ = model.encode(img2.unsqueeze(0).to(device))
            z1, z2 = mu1.squeeze(0), mu2.squeeze(0)
        else:
            z1 = model.encode(img1.unsqueeze(0).to(device)).squeeze(0)
            z2 = model.encode(img2.unsqueeze(0).to(device)).squeeze(0)

        alphas = torch.linspace(0, 1, n_steps)
        zs = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
        x_hat = model.decode(zs).cpu()

    fig, axes = plt.subplots(1, n_steps, figsize=(1.5 * n_steps, 2))
    for ax, img in zip(axes.ravel(), x_hat):
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.axis("off")
    plt.tight_layout()


def plot_pixel_histogram(dataset, n_images=200):
    loader = DataLoader(dataset, batch_size=n_images, shuffle=True)
    images, _ = next(iter(loader))
    plt.hist(images.flatten().numpy(), bins=50)
    plt.xlabel("pixel value")
    plt.ylabel("count")
    plt.yscale("log")


def plot_reconstructions(model, dataset, device, indices, is_vae=False):
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))
    with torch.no_grad():
        for col, idx in enumerate(indices):
            img, _ = dataset[idx]
            x = img.unsqueeze(0).to(device)
            if is_vae:
                x_hat, mu, logvar = model(x)
            else:
                x_hat, z = model(x)
            x_hat = x_hat.squeeze(0).cpu()

            axes[0, col].imshow(img.squeeze(0), cmap="gray")
            axes[0, col].axis("off")
            axes[1, col].imshow(x_hat.squeeze(0), cmap="gray")
            axes[1, col].axis("off")

    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("reconstruction")
    plt.tight_layout()
