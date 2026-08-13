from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.numerical.common import FEATURES, STATUS_CLASSES


SOURCE = ROOT.parent / "artifacts" / "numerical_synthetic" / "data" / "synthetic_numerical_20000.csv"
MODEL_PATH = ROOT / "models" / "numerical_synthetic_18_xgboost.joblib"
METRICS_PATH = ROOT / "results" / "numerical" / "synthetic_18_runtime_metrics.json"
TARGETS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
SEED = 42


def main() -> None:
    print(f"[NUMERICAL][Synthetic-only XGBoost 18-input] source={SOURCE}", flush=True)
    frame = pd.read_csv(SOURCE)
    indices = np.arange(len(frame))
    train_indices, holdout = train_test_split(
        indices, test_size=0.30, stratify=frame["Mental_Health_Status"], random_state=SEED
    )
    validation_indices, test_indices = train_test_split(
        holdout, test_size=0.50, stratify=frame.iloc[holdout]["Mental_Health_Status"], random_state=SEED
    )
    label_to_index = {label: index for index, label in enumerate(STATUS_CLASSES)}
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame["Mental_Health_Status"].map(label_to_index).to_numpy(dtype=np.int64)

    started = time.time()
    print("[NUMERICAL] Training classifier...", flush=True)
    classifier = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, n_jobs=-1, random_state=SEED, eval_metric="mlogloss",
    )
    classifier.fit(x[train_indices], y[train_indices], eval_set=[(x[validation_indices], y[validation_indices])], verbose=False)

    regressors = {}
    regression_r2 = {}
    for target in TARGETS:
        print(f"[NUMERICAL] Training regressor: {target}", flush=True)
        regressor = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.85,
            colsample_bytree=0.85, n_jobs=-1, random_state=SEED, objective="reg:squarederror",
        )
        regressor.fit(x[train_indices], frame[target].to_numpy()[train_indices], verbose=False)
        regressors[target] = regressor
        regression_r2[target] = float(r2_score(frame[target].to_numpy()[test_indices], regressor.predict(x[test_indices])))

    probabilities = classifier.predict_proba(x[test_indices])
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "protocol": "Synthetic-only, 18 frontend-compatible features, seeded 70/15/15 split",
        "accuracy": float(accuracy_score(y[test_indices], predictions)),
        "macro_f1": float(f1_score(y[test_indices], predictions, average="macro")),
        "weighted_f1": float(f1_score(y[test_indices], predictions, average="weighted")),
        "macro_roc_auc": float(roc_auc_score(y[test_indices], probabilities, multi_class="ovr", average="macro")),
        "regression_r2": regression_r2,
        "test_samples": int(len(test_indices)),
        "elapsed_seconds": time.time() - started,
    }
    joblib.dump(
        {"classifier": classifier, "regressors": regressors, "features": FEATURES, "classes": STATUS_CLASSES, "targets": TARGETS},
        MODEL_PATH,
        compress=3,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[NUMERICAL] Saved weights: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
