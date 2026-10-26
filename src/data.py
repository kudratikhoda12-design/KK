"""Synthetic pressure-mattress sensor data generator.

Why synthetic: public PMAP-style bed-pressure datasets are request-gated
(human-subjects data) and slow to obtain for a learning project. We generate
a programmatic stand-in that keeps the *structure* of the real problem:

  - a low-resolution 2D sensor grid (H x W taxels), one scalar reading per
    cell, roughly in [0, 1] after normalization
  - a body made of overlapping anisotropic Gaussian "blobs" (head, shoulders,
    torso/chest, pelvis, thighs, calves) whose position/size/intensity vary
    with posture and body size
  - an explicit, documented sensor noise model applied on top of the clean
    pressure field, so the denoising task (Phase 1) has a known ground truth
    and a stated noise assumption

Classes (posture): supine, prone, left_lateral, right_lateral.
Nuisance variables (not given to the model, used later for failure analysis):
body_size in {small, medium, large}, noise_level in {low, medium, high}.
"""
from __future__ import annotations

import numpy as np

H, W = 48, 24  # sensor grid resolution (rows x cols)
POSTURES = ["supine", "prone", "left_lateral", "right_lateral"]
BODY_SIZES = ["small", "medium", "large"]
NOISE_LEVELS = ["low", "medium", "high"]

_SIZE_SCALE = {"small": 0.82, "medium": 1.0, "large": 1.22}
_NOISE_PARAMS = {
    # (gaussian_sigma, dropout_prob, salt_prob, quant_levels)
    "low": (0.02, 0.01, 0.002, 64),
    "medium": (0.05, 0.03, 0.006, 32),
    "high": (0.09, 0.06, 0.012, 16),
}


def _gaussian_blob(yy, xx, cy, cx, sy, sx, amp):
    return amp * np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2) + ((xx - cx) ** 2) / (2 * sx ** 2)))


def _body_layout(posture: str, size_scale: float, rng: np.random.Generator):
    """Return a list of (cy, cx, sy, sx, amp) Gaussian blobs for one sample.

    Includes a *global* random placement offset (subject not perfectly
    centered/aligned on the mattress -- realistic for real sensor data) on
    top of per-blob jitter and amplitude noise. This is what makes the task
    non-trivial in raw pixel space: without it, posture classes occupy
    near-identical, cleanly separable pixel coordinates and a linear
    classifier on raw pixels trivially wins, which defeats the point of
    comparing it against learned representations.
    """
    jitter = lambda scale: rng.normal(0, scale)
    blobs = []

    # whole-body nuisance placement offset, independent of posture label
    global_dy = rng.normal(0, 2.6)
    global_dx = rng.normal(0, 1.8)

    # fractional vertical anchor positions along the body (head->feet)
    anchors = dict(head=0.07, shoulders=0.20, chest=0.37, pelvis=0.55, thigh=0.72, calf=0.90)

    if posture in ("supine", "prone"):
        cx_center = W / 2 + jitter(0.4) + global_dx
        width_mult = 1.0
        leg_offset = 2.2 * size_scale * rng.uniform(0.85, 1.15)
        chest_amp, pelvis_amp = (0.55, 0.85) if posture == "supine" else (0.85, 0.55)
        chest_amp += jitter(0.07)
        pelvis_amp += jitter(0.07)
    else:
        # side-lying: whole body shifts toward one half, narrower footprint
        side = -1.0 if posture == "left_lateral" else 1.0
        cx_center = W / 2 + side * (3.0 + jitter(0.5)) + global_dx
        width_mult = 0.62
        leg_offset = 1.1 * size_scale * rng.uniform(0.85, 1.15)
        chest_amp, pelvis_amp = 0.68 + jitter(0.07), 0.68 + jitter(0.07)

    for part, frac in anchors.items():
        cy = frac * H + jitter(1.1) + global_dy
        if part == "head":
            sy, sx, amp = 2.6 * size_scale, 2.2 * size_scale * width_mult, 0.45
            blobs.append((cy, cx_center, sy, sx, amp + jitter(0.06)))
        elif part == "shoulders":
            sy, sx, amp = 2.3 * size_scale, 5.2 * size_scale * width_mult, 0.60
            blobs.append((cy, cx_center, sy, sx, amp + jitter(0.06)))
        elif part == "chest":
            sy, sx, amp = 3.0 * size_scale, 4.0 * size_scale * width_mult, chest_amp
            blobs.append((cy, cx_center, sy, sx, amp))
        elif part == "pelvis":
            sy, sx, amp = 2.6 * size_scale, 3.8 * size_scale * width_mult, pelvis_amp
            blobs.append((cy, cx_center, sy, sx, amp))
        elif part == "thigh":
            sy, sx, amp = 3.4 * size_scale, 1.9 * size_scale * width_mult, 0.5
            blobs.append((cy, cx_center - leg_offset, sy, sx, amp + jitter(0.06)))
            blobs.append((cy, cx_center + leg_offset, sy, sx, amp + jitter(0.06)))
        elif part == "calf":
            sy, sx, amp = 3.0 * size_scale, 1.5 * size_scale * width_mult, 0.33
            blobs.append((cy, cx_center - leg_offset * 0.9, sy, sx, amp + jitter(0.04)))
            blobs.append((cy, cx_center + leg_offset * 0.9, sy, sx, amp + jitter(0.04)))
    return blobs


def render_clean(posture: str, body_size: str, rng: np.random.Generator) -> np.ndarray:
    size_scale = _SIZE_SCALE[body_size] * rng.uniform(0.88, 1.12)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    field = np.zeros((H, W), dtype=np.float32)
    for cy, cx, sy, sx, amp in _body_layout(posture, size_scale, rng):
        field += _gaussian_blob(yy, xx, cy, cx, sy, sx, amp)
    field = np.clip(field, 0.0, 1.5)
    field = field / 1.5
    return field.astype(np.float32)


def add_sensor_noise(clean: np.ndarray, noise_level: str, rng: np.random.Generator) -> np.ndarray:
    """Documented noise model: Gaussian electrical noise + dead-cell dropout
    + impulsive salt spikes (cross-talk) + ADC quantization."""
    sigma, p_drop, p_salt, levels = _NOISE_PARAMS[noise_level]
    noisy = clean + rng.normal(0, sigma, size=clean.shape).astype(np.float32)

    drop_mask = rng.random(clean.shape) < p_drop
    noisy[drop_mask] = 0.0

    salt_mask = rng.random(clean.shape) < p_salt
    noisy[salt_mask] = rng.uniform(0.8, 1.0, size=salt_mask.sum()).astype(np.float32)

    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = np.round(noisy * (levels - 1)) / (levels - 1)
    return noisy.astype(np.float32)


def sample_batch(n: int, seed: int, noise_levels=None, body_sizes=None, postures=None):
    """Generate n samples. Returns dict of arrays: clean, noisy, label, body_size, noise_level."""
    rng = np.random.default_rng(seed)
    postures = postures or POSTURES
    body_sizes = body_sizes or BODY_SIZES
    noise_levels = noise_levels or NOISE_LEVELS

    clean = np.zeros((n, H, W), dtype=np.float32)
    noisy = np.zeros((n, H, W), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    bsize = np.empty(n, dtype=object)
    nlevel = np.empty(n, dtype=object)

    for i in range(n):
        posture = postures[rng.integers(0, len(postures))]
        bs = body_sizes[rng.integers(0, len(body_sizes))]
        nl = noise_levels[rng.integers(0, len(noise_levels))]
        c = render_clean(posture, bs, rng)
        nz = add_sensor_noise(c, nl, rng)
        clean[i] = c
        noisy[i] = nz
        labels[i] = POSTURES.index(posture)
        bsize[i] = bs
        nlevel[i] = nl

    return dict(clean=clean, noisy=noisy, label=labels, body_size=bsize, noise_level=nlevel)


def label_name(idx: int) -> str:
    return POSTURES[idx]


if __name__ == "__main__":
    batch = sample_batch(8, seed=0)
    print({k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in batch.items()})
    print("label counts:", np.bincount(batch["label"]))
