"""
MAI/IDL SS26 - VAE demo.

MG 30/6/2026
"""

import torch
import torch.nn as nn


class AE(nn.Module):
    """Plain (non-variational) autoencoder. CNN encoder/decoder, 2D bottleneck."""

    def __init__(self, latent_dim=2):
        super().__init__()

        # encoder: 28x28 -> 14x14 -> 7x7 -> latent_dim
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_fc = nn.Linear(32 * 7 * 7, latent_dim)
        self.relu = nn.ReLU()

        # decoder: latent_dim -> 7x7 -> 14x14 -> 28x28
        self.dec_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.enc_conv1(x))
        h = self.relu(self.enc_conv2(h))
        h = h.flatten(1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.relu(self.dec_fc(z))
        h = h.view(-1, 32, 7, 7)
        h = self.relu(self.dec_conv1(h))
        x_hat = self.sigmoid(self.dec_conv2(h))
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class VAE(nn.Module):
    """Variational autoencoder. Same CNN backbone as AE, but encoder outputs
    a mean and log-variance, and we sample the latent code via the
    reparameterization trick."""

    def __init__(self, latent_dim=2):
        super().__init__()

        # encoder: 28x28 -> 14x14 -> 7x7 -> (mu, logvar)
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.enc_fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
        self.relu = nn.ReLU()

        # decoder: latent_dim -> 7x7 -> 14x14 -> 28x28 (identical to AE decoder)
        self.dec_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.enc_conv1(x))
        h = self.relu(self.enc_conv2(h))
        h = h.flatten(1)
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = self.relu(self.dec_fc(z))
        h = h.view(-1, 32, 7, 7)
        h = self.relu(self.dec_conv1(h))
        x_hat = self.sigmoid(self.dec_conv2(h))
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
