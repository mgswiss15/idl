"""
MAI/IDL SS26 - VAE demo.

MG 30/6/2026
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Conv trunk shared by AE and VAE: 28x28 image -> flattened feature vector.
    Stops before any latent projection -- that part differs between AE and VAE."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)   # 28 -> 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 14 -> 7
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = h.flatten(1)  # (batch, 32*7*7)
        return h


class Decoder(nn.Module):
    """Conv trunk shared by AE and VAE: latent vector -> 28x28 image.
    Identical for both -- only what feeds into it differs."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 7 -> 14
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)   # 14 -> 28
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc(z))
        h = h.view(-1, 32, 7, 7)
        h = self.relu(self.deconv1(h))
        x_hat = self.sigmoid(self.deconv2(h))
        return x_hat


class AE(nn.Module):
    """Plain (non-variational) autoencoder: a single latent vector per image."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder()
        self.fc_latent = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc_latent(h)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class VAE(nn.Module):
    """Variational autoencoder: same encoder/decoder trunks as the AE, but the
    encoder produces a mean and log-variance, and the latent code is sampled
    via the reparameterization trick."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder()
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
