"""Loader for the real PmatData dataset (Experiment I): "A Pressure Map
Dataset for In-bed Posture Classification" (Baran Pouyan et al., IEEE BHI
2017), PhysioNet, open access, DOI 10.13026/C2W09K.

Experiment I: 13 subjects x 17 labeled positions, Vista Medical FSA
SoftFlex 2048 mat, 32x64 taxel grid at 1Hz, ~70-120 frames per
subject/position file. Position -> (posture, variant) mapping below is
transcribed directly from the dataset's own experiment-info.docx posture
table (Index/Posture/Bed Inclination/Body-roll columns).

Note: the real dataset has no "prone" posture (none was recorded), so the
real-data study uses 3 posture classes (supine/left/right) rather than the
4 used in the synthetic study (supine/prone/left_lateral/right_lateral).
Bed inclination and body-roll wedges/pillow are recorded as a "variant"
nuisance label, analogous to body_size in the synthetic study.
"""
from __future__ import annotations

import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "external_data", "pmd")
EXP1_DIR = os.path.join(ROOT, "experiment-i")

H, W = 64, 32  # real sensor grid, reshaped to (length-along-body, width)
RAW_CLIP = 1024.0  # normalization constant (see scripts/09_prepare_real_data.py for justification)

POSTURES = ["supine", "left", "right"]

# position index -> (posture, variant), transcribed from experiment-info.docx
POSITION_LABELS = {
    1: ("supine", "flat"),
    2: ("right", "flat"),
    3: ("left", "flat"),
    4: ("right", "wedge30"),
    5: ("right", "wedge60"),
    6: ("left", "wedge30"),
    7: ("left", "wedge60"),
    8: ("supine", "flat"),
    9: ("supine", "flat"),
    10: ("supine", "flat"),
    11: ("supine", "flat"),
    12: ("supine", "flat"),
    13: ("right", "fetus"),
    14: ("left", "fetus"),
    15: ("supine", "incline30"),
    16: ("supine", "incline45"),
    17: ("supine", "incline60"),
}

TEST_SUBJECTS = [2, 7, 12]  # held out entirely: tests generalization to an unseen person
TRAIN_SUBJECTS = [s for s in range(1, 14) if s not in TEST_SUBJECTS]

SUBJECT_INFO = {  # subject_id -> (age, height_cm, weight_kg), from subject-info-i.csv
    1: (19, 175, 87), 2: (23, 183, 85), 3: (23, 183, 100), 4: (24, 177, 70),
    5: (24, 172, 66), 6: (26, 169, 83), 7: (27, 179, 96), 8: (27, 186, 63),
    9: (30, 174, 74), 10: (30, 174, 79), 11: (30, 176, 91), 12: (33, 170, 78),
    13: (34, 174, 74),
}


def load_subject_position(subject_id: int, position: int) -> np.ndarray:
    path = os.path.join(EXP1_DIR, f"S{subject_id}", f"{position}.txt")
    arr = np.loadtxt(path)  # (n_frames, 2048)
    frames = arr.reshape(arr.shape[0], H, W)
    frames = np.clip(frames, 0, RAW_CLIP) / RAW_CLIP
    return frames.astype(np.float32)


def load_subjects(subject_ids):
    """Returns dict of arrays: clean, label, variant, subject_id, posture_name."""
    all_frames, labels, variants, subjects = [], [], [], []
    for sid in subject_ids:
        for pos, (posture, variant) in POSITION_LABELS.items():
            frames = load_subject_position(sid, pos)
            all_frames.append(frames)
            n = frames.shape[0]
            labels.extend([POSTURES.index(posture)] * n)
            variants.extend([variant] * n)
            subjects.extend([sid] * n)
    clean = np.concatenate(all_frames, axis=0)
    return dict(
        clean=clean,
        label=np.array(labels, dtype=np.int64),
        variant=np.array(variants, dtype=object),
        subject_id=np.array(subjects, dtype=np.int64),
    )


if __name__ == "__main__":
    d = load_subjects([1])
    print({k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in d.items()})
    print("label counts:", np.bincount(d["label"]))
