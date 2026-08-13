"""train original implementation for the curated submission repository."""
from __future__ import annotations

import time

import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from src.numerical.common import (
    FEATURES, REGRESSION_TARGETS, SEED, STATUS_CLASSES, arrays_from_dataframe,
    dump_json, load_dataframe, load_fixed_split, root,
)
from src.numerical.metrics import regression_metrics
from src.speech.metrics_utils import compute_metrics

R = root()
OUT_DIR = R / "artifacts" / "numerical"


def main():
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    train_idx, val_idx, test_idx = load_fixed_split()
    x, y_status, y_reg = arrays_from_dataframe(df)

    print("[NUMERICAL][classical] training CatBoostClassifier ...", flush=True)
    classifier = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.06, loss_function="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=False,
    )
    classifier.fit(x[train_idx], y_status[train_idx], eval_set=(x[val_idx], y_status[val_idx]), use_best_model=True)
    test_proba = classifier.predict_proba(x[test_idx])
    classification_metrics = compute_metrics(y_status[test_idx], test_proba, STATUS_CLASSES)
    classification_metrics.update({"model": "CatBoostClassifier", "best_iteration": classifier.get_best_iteration()})
    joblib.dump({"model": classifier, "classes": STATUS_CLASSES, "features": FEATURES}, OUT_DIR / "classical_baseline_classifier.joblib")

    print("[NUMERICAL][classical] training CatBoostRegressor (one per target) ...", flush=True)
    regressors = {}
    test_predictions = np.zeros((len(test_idx), len(REGRESSION_TARGETS)), dtype=np.float32)
    for i, target in enumerate(REGRESSION_TARGETS):
        model = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.06, loss_function="RMSE", random_seed=SEED, verbose=False)
        model.fit(x[train_idx], y_reg[train_idx, i], eval_set=(x[val_idx], y_reg[val_idx, i]), use_best_model=True)
        test_predictions[:, i] = model.predict(x[test_idx])
        regressors[target] = model
    joblib.dump({"models": regressors, "targets": REGRESSION_TARGETS, "features": FEATURES}, OUT_DIR / "classical_baseline_regressors.joblib")
    regression_test_metrics = regression_metrics(y_reg[test_idx], test_predictions, REGRESSION_TARGETS)

    payload = {
        "classification": classification_metrics, "regression": regression_test_metrics,
        "training_seconds": time.time() - started, "features": FEATURES,
        "targets_excluded_from_classification_inputs": REGRESSION_TARGETS,
    }
    dump_json(OUT_DIR / "classical_baseline_metrics.json", payload)

    print("[NUMERICAL][classical] BASELINE COMPLETE", flush=True)
    print({
        "accuracy": classification_metrics["accuracy"], "macro_f1": classification_metrics["f1_macro"],
        "weighted_f1": classification_metrics["f1_weighted"], "roc_auc_macro_ovr": classification_metrics["roc_auc_macro_ovr"],
        "regression": {t: {"mae": regression_test_metrics[t]["mae"], "r2": regression_test_metrics[t]["r2"]} for t in REGRESSION_TARGETS},
        "training_seconds": payload["training_seconds"],
    })


if __name__ == "__main__":
    main()




