"""Shared model components.

A single conv `Backbone` is reused everywhere so that the few-shot ablation
(denoising-pretrained vs contrastively-pretrained vs random-init encoder) is
an apples-to-apples comparison of the *same* architecture with different
weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_CH = 64  # channels at the bottleneck


def _conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


def _deconv_block(c_in, c_out, final=False):
    layers = [nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1)]
    if not final:
        layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Backbone(nn.Module):
    """1x48x24 -> 64x6x3 spatial feature map."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, LATENT_CH),
        )

    def forward(self, x):
        return self.net(x)

    def pooled(self, x):
        feat = self.net(x)
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, LATENT_CH)


class Decoder(nn.Module):
    """64x6x3 -> 1x48x24."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _deconv_block(LATENT_CH, 32),
            _deconv_block(32, 16),
            _deconv_block(16, 1, final=True),
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.backbone(x)
        return self.decoder(z), z


class ProjectionHead(nn.Module):
    """Pooled backbone features -> contrastive embedding space."""

    def __init__(self, in_dim=LATENT_CH, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class LinearProbe(nn.Module):
    def __init__(self, in_dim=LATENT_CH, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CNNFromScratch(nn.Module):
    """Same backbone architecture trained end-to-end, randomly initialized,
    used as the 'CNN trained from scratch' baseline under few labels."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = Backbone()
        self.fc = nn.Linear(LATENT_CH, num_classes)

    def forward(self, x):
        feat = self.backbone.pooled(x)
        return self.fc(feat)
