"""Download (with integrity check) and load the Polish bankruptcy ratios."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from . import config


def download(dest: Path = config.RAW_DIR, force: bool = False) -> Path:
    """Fetch and verify the UCI archive, extract the ARFF files into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    if not force and all((dest / f"{h}year.arff").exists() for h in config.ALL_HORIZONS):
        return dest
    with urllib.request.urlopen(config.DATASET_URL, timeout=120) as resp:
        blob = resp.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != config.DATASET_SHA256:
        raise RuntimeError(f"Dataset checksum mismatch: got {digest}, expected {config.DATASET_SHA256}")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if "/" in name or ".." in name:  # flat archive expected; refuse anything else
                raise RuntimeError(f"Unexpected path in archive: {name}")
            zf.extract(name, dest)
    return dest


def load_horizon(horizon: int = config.PRIMARY_HORIZON, raw_dir: Path = config.RAW_DIR,
                 dedupe: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X, y) for one forecast horizon. Columns are ``Attr1..Attr64``.

    Exact duplicate feature rows are dropped (none carry conflicting labels), so a
    firm cannot land in both the train and the test split.
    """
    path = raw_dir / f"{horizon}year.arff"
    if not path.exists():
        download(raw_dir)
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    y = df.pop("class").map(lambda v: int(v.decode() if isinstance(v, bytes) else v)).to_numpy()
    X = df.astype(float)
    if dedupe:
        keep = ~X.duplicated(keep="first").to_numpy()
        X, y = X.loc[keep].reset_index(drop=True), y[keep]
    return X, y


if __name__ == "__main__":
    print(f"Data ready in {download()}")
