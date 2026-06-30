"""
MAI/IDL SS26 - VAE demo.

MG 30/6/2026
"""

import torch
import torch.nn.functional as F


def train_ae(model, train_loader, optimizer, num_epochs, device):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            x_hat, z = model(images)

            # reconstruction loss only (L2), summed over pixels, averaged over batch
            loss = F.mse_loss(x_hat, images, reduction="sum") / images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

    return model, train_losses


def train_vae(model, train_loader, optimizer, num_epochs, device, recon_loss="l2"):
    train_losses = []
    recon_losses = []
    kl_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        for images, _ in train_loader:
            images = images.to(device)
            x_hat, mu, logvar = model(images)

            # reconstruction term: -log p(x|z), summed over pixels, averaged over batch
            if recon_loss == "l2":
                recon = F.mse_loss(x_hat, images, reduction="sum") / images.size(0)
            elif recon_loss == "bce":
                recon = F.binary_cross_entropy(x_hat, images, reduction="sum") / images.size(0)
            else:
                raise ValueError(f"Unknown recon_loss: {recon_loss}")

            # KL(q(z|x) || N(0,I)), summed over latent dims, averaged over batch
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)

            loss = recon + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        n = len(train_loader)
        train_losses.append(epoch_loss / n)
        recon_losses.append(epoch_recon / n)
        kl_losses.append(epoch_kl / n)
        print(f"Epoch {epoch+1}: loss={epoch_loss/n:.4f}, recon={epoch_recon/n:.4f}, kl={epoch_kl/n:.4f}")

    return model, train_losses, recon_losses, kl_losses
