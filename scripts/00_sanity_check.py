"""Quick visual sanity check of the synthetic data generator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import render_clean, add_sensor_noise, POSTURES, BODY_SIZES, NOISE_LEVELS

rng = np.random.default_rng(42)
fig, axes = plt.subplots(4, 4, figsize=(8, 12))
for i, posture in enumerate(POSTURES):
    clean = render_clean(posture, "medium", rng)
    noisy_low = add_sensor_noise(clean, "low", rng)
    noisy_high = add_sensor_noise(clean, "high", rng)
    small = render_clean(posture, "small", rng)
    axes[i, 0].imshow(clean, cmap="inferno", vmin=0, vmax=1)
    axes[i, 0].set_title(f"{posture}\nclean")
    axes[i, 1].imshow(noisy_low, cmap="inferno", vmin=0, vmax=1)
    axes[i, 1].set_title("noisy (low)")
    axes[i, 2].imshow(noisy_high, cmap="inferno", vmin=0, vmax=1)
    axes[i, 2].set_title("noisy (high)")
    axes[i, 3].imshow(small, cmap="inferno", vmin=0, vmax=1)
    axes[i, 3].set_title("clean (small body)")
    for ax in axes[i]:
        ax.axis("off")
plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/00_sanity_check.png", dpi=130)
print("saved reports/figures/00_sanity_check.png")
