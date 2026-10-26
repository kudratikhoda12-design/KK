"""Label-preserving augmentations for contrastive pretraining.

Note: horizontal flip is deliberately NOT used — it would map
left_lateral <-> right_lateral and silently corrupt the posture label, which
is exactly the kind of augmentation bug this project wants to avoid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def add_noise(x, sigma=0.06):
    return x + torch.randn_like(x) * sigma


def random_translate(x, max_shift=2):
    B, C, H, W = x.shape
    out = torch.zeros_like(x)
    for i in range(B):
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        out[i] = torch.roll(x[i], shifts=(dy, dx), dims=(1, 2))
    return out


def random_dropout(x, p=0.05):
    mask = (torch.rand_like(x) > p).float()
    return x * mask


def random_intensity_scale(x, lo=0.85, hi=1.15):
    B = x.shape[0]
    scale = torch.empty(B, 1, 1, 1, device=x.device).uniform_(lo, hi)
    return (x * scale).clamp(0, 1)


def augment_view(x):
    x = random_translate(x, max_shift=2)
    x = random_intensity_scale(x)
    x = add_noise(x, sigma=0.06)
    x = random_dropout(x, p=0.04)
    return x.clamp(0, 1)
