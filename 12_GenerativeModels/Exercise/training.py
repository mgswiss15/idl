"""
MAI/IDL SS26 - VAE demo.

MG 30/6/2026
"""

import torch
import torch.nn as nn


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, is_vae=False):
    """Single training loop for both AE (is_vae=False) and VAE (is_vae=True).
    criterion is the reconstruction loss, e.g. nn.MSELoss(reduction='sum') or
    nn.BCELoss(reduction='sum'). For the AE the KL term is simply zero throughout."""
    train_losses = []
    recon_losses = []
    kl_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        for images, _ in train_loader:
            images = images.to(device)

            if is_vae:
                x_hat, mu, logvar = model(images)
                recon = criterion(x_hat, images) / images.size(0)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
            else:
                x_hat, z = model(images)
                recon = criterion(x_hat, images) / images.size(0)
                kl = torch.zeros(1, device=device)

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


def evaluate_reconstruction(model, test_loader, criterion, device, is_vae=False):
    """Average reconstruction error on held-out data, using the same
    sum-over-pixels / mean-over-batch convention as training."""
    model.eval()
    total_recon = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            if is_vae:
                x_hat, mu, logvar = model(images)
            else:
                x_hat, z = model(images)
            recon = criterion(x_hat, images) / images.size(0)
            total_recon += recon.item()
    return total_recon / len(test_loader)


def train_probe(z_train, y_train, z_test, y_test, num_epochs, device, lr=0.1):
    """Train a simple linear classifier (logistic regression) on top of frozen
    latent codes, to probe how much class information they carry."""
    probe = nn.Linear(z_train.size(1), 10).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    z_train, y_train = z_train.to(device), y_train.to(device)
    z_test, y_test = z_test.to(device), y_test.to(device)

    for epoch in range(num_epochs):
        probe.train()
        logits = probe(z_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(z_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return probe, accuracy
