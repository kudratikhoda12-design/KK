"""Parses the raw PmatData Experiment I text files (run
scripts/08_download_real_data.sh first) and caches them into the same
split structure used throughout the project, prefixed `real_*`.

Split design (subject-level, not frame-level): 3 of 13 subjects
(real_data.TEST_SUBJECTS) are held out *entirely* for `real_test_eval` --
none of their frames are seen during pretraining, denoising training, or
few-shot label selection. This tests generalization to an unseen person's
body, which is a materially harder and more honest test than the
frame-level random splits used in the synthetic study (where the same
"body" distribution, just not the same exact sample, could appear in both
train and test).

Since PmatData frames are single real observations (no paired clean/noisy
version), the Phase-1 denoising target uses the raw frame as "clean" and
applies the *same* documented synthetic noise model from src/data.py on
top of it to produce the "noisy" input -- consistent methodology to the
synthetic study, now validated against a real signal.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from src.real_data import load_subjects, TRAIN_SUBJECTS, TEST_SUBJECTS, POSTURES
from src.data import add_sensor_noise

os.makedirs("data", exist_ok=True)

print(f"train subjects: {TRAIN_SUBJECTS}")
print(f"test subjects (held out entirely): {TEST_SUBJECTS}")

train_data = load_subjects(TRAIN_SUBJECTS)
test_data = load_subjects(TEST_SUBJECTS)

print(f"train frames: {train_data['clean'].shape[0]}  label counts: {np.bincount(train_data['label'])}")
print(f"test frames:  {test_data['clean'].shape[0]}  label counts: {np.bincount(test_data['label'])}")

rng = np.random.default_rng(0)

# denoise_train/val: frame-level split within train subjects only
n_train = train_data["clean"].shape[0]
perm = rng.permutation(n_train)
n_val = int(0.15 * n_train)
val_idx, tr_idx = perm[:n_val], perm[n_val:]


def make_noisy(clean_arr, seed):
    r = np.random.default_rng(seed)
    noise_levels = ["low", "medium", "high"]
    out = np.zeros_like(clean_arr)
    for i in range(clean_arr.shape[0]):
        nl = noise_levels[r.integers(0, 3)]
        out[i] = add_sensor_noise(clean_arr[i], nl, r)
    return out


denoise_train_clean = train_data["clean"][tr_idx]
denoise_val_clean = train_data["clean"][val_idx]
np.savez_compressed(
    "data/real_denoise_train.npz",
    clean=denoise_train_clean, noisy=make_noisy(denoise_train_clean, seed=1),
)
np.savez_compressed(
    "data/real_denoise_val.npz",
    clean=denoise_val_clean, noisy=make_noisy(denoise_val_clean, seed=2),
)

# pretrain_unlabeled + fewshot_pool: the full train-subject pool (noisy version used
# everywhere downstream, consistent with "this is what the sensor actually reports")
train_noisy = make_noisy(train_data["clean"], seed=3)
np.savez_compressed(
    "data/real_pretrain_unlabeled.npz",
    clean=train_data["clean"], noisy=train_noisy,
    label=train_data["label"], variant=train_data["variant"], subject_id=train_data["subject_id"],
)
np.savez_compressed(
    "data/real_fewshot_pool.npz",
    clean=train_data["clean"], noisy=train_noisy,
    label=train_data["label"], variant=train_data["variant"], subject_id=train_data["subject_id"],
)

# test_eval: held-out subjects entirely
test_noisy = make_noisy(test_data["clean"], seed=4)
np.savez_compressed(
    "data/real_test_eval.npz",
    clean=test_data["clean"], noisy=test_noisy,
    label=test_data["label"], variant=test_data["variant"], subject_id=test_data["subject_id"],
)

print("saved data/real_*.npz")
print("posture classes:", POSTURES)
