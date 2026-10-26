"""Generate and cache all dataset splits used by the rest of the pipeline.

Splits (disjoint RNG seeds -> no leakage):
  - denoise_train / denoise_val : clean+noisy pairs for Phase 1 (autoencoder)
  - pretrain_unlabeled          : noisy-only pool for Phase 2 contrastive pretraining
  - fewshot_pool                : labeled pool to draw k-shot subsets from
  - test_eval                   : held-out evaluation set (also used for failure analysis)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data import sample_batch

os.makedirs("data", exist_ok=True)

SPLITS = {
    "denoise_train": (2000, 1),
    "denoise_val": (500, 2),
    "pretrain_unlabeled": (3000, 3),
    "fewshot_pool": (2400, 4),  # 600/class on average, plenty for k<=10 shots many trials
    "test_eval": (800, 5),  # 200/class
}

for name, (n, seed) in SPLITS.items():
    batch = sample_batch(n, seed=seed)
    path = f"data/{name}.npz"
    np.savez_compressed(
        path,
        clean=batch["clean"],
        noisy=batch["noisy"],
        label=batch["label"],
        body_size=batch["body_size"].astype(str),
        noise_level=batch["noise_level"].astype(str),
    )
    counts = np.bincount(batch["label"], minlength=4)
    print(f"{name}: n={n} seed={seed} label_counts={counts.tolist()} -> {path}")
