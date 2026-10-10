"""Data layer: SQL extraction / feature engineering (DuckDB) + model preprocessing."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def _sql(name: str) -> str:
    return (C.SQL_DIR / name).read_text()


def extract_cohort(raw_csv: str, n_sample: int = C.N_SAMPLE) -> pd.DataFrame:
    """Run 01_extract_cohort.sql against the raw LendingClub CSV (only needed once)."""
    con = duckdb.connect()
    query = _sql("01_extract_cohort.sql").format(raw_path=raw_csv, n_sample=n_sample)
    df = con.sql(query).df()
    C.DATA_DIR.mkdir(exist_ok=True, parents=True)
    df.to_parquet(C.SAMPLE_PATH, index=False)
    return df


def build_model_table() -> pd.DataFrame:
    """Run the SQL pipeline (02 survival table, 03 feature engineering) on the sample."""
    con = duckdb.connect()
    con.sql(f"CREATE TABLE raw_sample AS SELECT * FROM read_parquet('{C.SAMPLE_PATH}')")
    for f in ["02_build_survival_table.sql", "03_feature_engineering.sql"]:
        con.execute(_sql(f))
    df = con.sql("SELECT * FROM model_table ORDER BY id").df()
    df.to_parquet(C.MODEL_TABLE_PATH, index=False)
    return df


def load_model_table() -> pd.DataFrame:
    if not C.MODEL_TABLE_PATH.exists():
        return build_model_table()
    return pd.read_parquet(C.MODEL_TABLE_PATH)


class Preprocessor:
    """Fit-on-train preprocessing: impute -> log1p -> winsorise -> standardise -> one-hot.

    Everything (medians, clip bounds, means/stds, category levels) is learned on the
    training split only to avoid leakage into validation/test.
    """

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = self._numeric(df)
        self.medians = X.median()
        X = X.fillna(self.medians)
        self.lo, self.hi = X.quantile(0.005), X.quantile(0.995)
        X = X.clip(self.lo, self.hi, axis=1)
        self.mean, self.std = X.mean(), X.std().replace(0, 1)
        self.levels = {c: sorted(df[c].dropna().unique()) for c in C.CATEGORICAL}
        return self

    @staticmethod
    def _numeric(df, cols=None):
        X = df[cols or C.NUMERIC].astype(float).copy()
        for c in X.columns.intersection(C.LOG1P):
            X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._numeric(df).fillna(self.medians).clip(self.lo, self.hi, axis=1)
        X = (X - self.mean) / self.std
        for c in C.BINARY:
            X[c] = df[c].astype(float).values
        for c, levels in self.levels.items():           # drop-first dummy coding
            for lv in levels[1:]:
                X[f"{c}={lv}"] = (df[c] == lv).astype(float).values
        return X

    @property
    def feature_names(self):
        return list(C.NUMERIC) + list(C.BINARY) + [f"{c}={lv}" for c, l in self.levels.items() for lv in l[1:]]


def split(df: pd.DataFrame, seed: int = C.SEED):
    """60 / 20 / 20 train / validation / test split, stratified on the event flag."""
    train, temp = train_test_split(df, test_size=0.4, stratify=df[C.EVENT], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[C.EVENT], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def prepare(seed: int = C.SEED):
    """Return preprocessed (X, t, e) arrays for train/val/test plus the fitted preprocessor."""
    df = load_model_table()
    tr, va, te = split(df, seed)
    pp = Preprocessor().fit(tr)
    out = {}
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        out[name] = dict(
            X=pp.transform(part),
            t=part[C.DURATION].to_numpy(float),
            e=part[C.EVENT].to_numpy(int),
            raw=part,
        )
    return out, pp


def z_to_raw(pp: Preprocessor, feature: str, z):
    """Map a standardised value back to the original unit (undo z-score and log1p)."""
    x = np.asarray(z) * pp.std[feature] + pp.mean[feature]
    return np.expm1(x) if feature in C.LOG1P else x
