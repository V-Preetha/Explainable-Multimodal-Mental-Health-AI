"""dataset regimes implementation for the curated submission repository."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.numerical.common import load_fixed_split
from src.numerical.synthetic_common import (
    ALL_FEATURES, DERIVED_FEATURES, NEW_RAW_FEATURES, ORIGINAL_FEATURES,
    REGRESSION_TARGETS, SEED, STATUS_TO_INDEX, data_dir, data_root,
)

IMPUTED_FEATURES = NEW_RAW_FEATURES + DERIVED_FEATURES


def load_original_dataframe() -> pd.DataFrame:
    df = pd.read_csv(data_root() / "mental_health_multimodal.csv")
    df["source"] = "original"
    for col in IMPUTED_FEATURES:
        df[col] = np.nan
    return df


def load_synthetic_dataframe() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "synthetic_numerical_20000.csv")


def split_synthetic_indices(df: pd.DataFrame, seed: int = SEED, val_fraction: float = 0.15, test_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df["Mental_Health_Status"].to_numpy()
    all_idx = np.arange(len(df))
    train_idx, holdout_idx = train_test_split(all_idx, test_size=val_fraction + test_fraction, stratify=y, random_state=seed)
    relative_val = val_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=1 - relative_val, stratify=y[holdout_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def _xy(df: pd.DataFrame, idx: np.ndarray) -> dict:
    rows = df.iloc[idx]
    return {
        "X": rows[ALL_FEATURES].to_numpy(dtype=np.float64),
        "y_status": rows["Mental_Health_Status"].map(STATUS_TO_INDEX).to_numpy(dtype=np.int64),
        "y_reg": rows[REGRESSION_TARGETS].to_numpy(dtype=np.float64),
        "source": rows["source"].to_numpy() if "source" in rows.columns else np.repeat("synthetic", len(rows)),
        "row_index": rows.index.to_numpy(),
    }


def build_n1_synthetic_only(seed: int = SEED) -> dict:
    synthetic_df = load_synthetic_dataframe()
    train_idx, val_idx, test_idx = split_synthetic_indices(synthetic_df, seed)
    return {
        "name": "N1_synthetic_only",
        "features": ALL_FEATURES,
        "train": _xy(synthetic_df, train_idx),
        "validation": _xy(synthetic_df, val_idx),
        "test": _xy(synthetic_df, test_idx),
    }


def build_n2_combined(seed: int = SEED) -> dict:
    original_df = load_original_dataframe()
    synthetic_df = load_synthetic_dataframe()
    orig_train_idx, orig_val_idx, orig_test_idx = load_fixed_split()
    syn_train_idx, syn_val_idx, syn_test_idx = split_synthetic_indices(synthetic_df, seed)

    # Impute the 53 synthetic-only columns for ALL original-provenance rows
    # using the median computed from the synthetic TRAINING split only (no
    # val/test statistics leak in); the same constant is reused for original
    # train/validation/test alike so it carries zero row-specific information.
    impute_values = synthetic_df.iloc[syn_train_idx][IMPUTED_FEATURES].median()
    original_df_imputed = original_df.copy()
    original_df_imputed[IMPUTED_FEATURES] = original_df_imputed[IMPUTED_FEATURES].fillna(impute_values)

    orig_train, orig_val, orig_test = _xy(original_df_imputed, orig_train_idx), _xy(original_df_imputed, orig_val_idx), _xy(original_df_imputed, orig_test_idx)
    syn_train, syn_val, syn_test = _xy(synthetic_df, syn_train_idx), _xy(synthetic_df, syn_val_idx), _xy(synthetic_df, syn_test_idx)

    def concat(a: dict, b: dict) -> dict:
        return {
            "X": np.concatenate([a["X"], b["X"]], axis=0),
            "y_status": np.concatenate([a["y_status"], b["y_status"]], axis=0),
            "y_reg": np.concatenate([a["y_reg"], b["y_reg"]], axis=0),
            "source": np.concatenate([a["source"], b["source"]], axis=0),
        }

    return {
        "name": "N2_combined_original_plus_synthetic",
        "features": ALL_FEATURES,
        "imputed_features": IMPUTED_FEATURES,
        "impute_values": impute_values.to_dict(),
        "train": concat(orig_train, syn_train),
        "validation": concat(orig_val, syn_val),
        "test_original_real": orig_test,        # PRIMARY evaluation split
        "test_synthetic_holdout": syn_test,      # secondary only, never substituted for the primary
        # Original-provenance rows only, split-for-split (same fixed
        # 2800/600/600 indices) -- used to export fusion-compatible
        # embeddings anchored on the real participant rows, not the
        # synthetic ones (synthetic rows have no corresponding face/speech
        # partners to align with in the multimodal fusion stage).
        "original_only": {"train": orig_train, "validation": orig_val, "test": orig_test},
    }




