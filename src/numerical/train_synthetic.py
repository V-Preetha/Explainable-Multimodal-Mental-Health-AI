"""train synthetic implementation for the curated submission repository."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from src.numerical.metrics import regression_metrics
from src.numerical.synthetic_common import (
    REGRESSION_TARGETS, SEED, STATUS_CLASSES, metrics_dir, models_dir,
)
from src.numerical.dataset_regimes import build_n1_synthetic_only, build_n2_combined

CLASSIFIER_FACTORIES = {
    "CatBoost": lambda: CatBoostClassifier(iterations=400, depth=6, learning_rate=0.06, loss_function="MultiClass", random_seed=SEED, verbose=False),
    "XGBoost": lambda: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, n_jobs=-1, random_state=SEED, eval_metric="mlogloss"),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
}
REGRESSOR_FACTORIES = {
    "CatBoost": lambda: CatBoostRegressor(iterations=400, depth=6, learning_rate=0.06, random_seed=SEED, verbose=False),
    "XGBoost": lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, n_jobs=-1, random_state=SEED),
    "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=400, n_jobs=-1, random_state=SEED),
}


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    pred = y_prob.argmax(axis=1)
    n_classes = len(STATUS_CLASSES)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(n_classes))).tolist(),
        "per_class": classification_report(y_true, pred, labels=list(range(n_classes)), target_names=STATUS_CLASSES, output_dict=True, zero_division=0),
    }
    try:
        metrics["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(n_classes))))
        metrics["roc_auc_weighted_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=list(range(n_classes))))
    except ValueError:
        metrics["roc_auc_macro_ovr"] = metrics["roc_auc_weighted_ovr"] = float("nan")
    return metrics


def run_classifier(name: str, regime_name: str, train: dict, val: dict, test_sets: dict[str, dict]):
    model = CLASSIFIER_FACTORIES[name]()
    started = time.time()
    if name == "CatBoost":
        model.fit(train["X"], train["y_status"], eval_set=(val["X"], val["y_status"]), use_best_model=True)
    else:
        model.fit(train["X"], train["y_status"])
    result = {"model": name, "regime": regime_name, "seconds": time.time() - started, "test_results": {}}
    for test_name, test in test_sets.items():
        result["test_results"][test_name] = classification_metrics(test["y_status"], model.predict_proba(test["X"]))
    print(f"[CLASSICAL] {regime_name}/{name} classifier: {result['seconds']:.1f}s | " + " | ".join(f"{k}_acc={v['accuracy']:.3f}" for k, v in result["test_results"].items()), flush=True)
    return result, model


def run_regressors(name: str, regime_name: str, train: dict, val: dict, test_sets: dict[str, dict]):
    started = time.time()
    models = {}
    for i, target in enumerate(REGRESSION_TARGETS):
        model = REGRESSOR_FACTORIES[name]()
        if name == "CatBoost":
            model.fit(train["X"], train["y_reg"][:, i], eval_set=(val["X"], val["y_reg"][:, i]), use_best_model=True)
        else:
            model.fit(train["X"], train["y_reg"][:, i])
        models[target] = model
    result = {"model": name, "regime": regime_name, "seconds": time.time() - started, "test_results": {}}
    for test_name, test in test_sets.items():
        preds = np.column_stack([models[t].predict(test["X"]) for t in REGRESSION_TARGETS])
        result["test_results"][test_name] = regression_metrics(test["y_reg"], preds, REGRESSION_TARGETS)
    print(f"[CLASSICAL] {regime_name}/{name} regressors: {result['seconds']:.1f}s", flush=True)
    return result, models


def try_tabpfn(train: dict, test_sets: dict[str, dict], max_train_rows: int = 2000, max_eval_rows: int = 2000) -> dict | None:
    try:
        from tabpfn import TabPFNClassifier
    except Exception as exc:
        print(f"[CLASSICAL][TabPFN] not usable, skipping: {exc}", flush=True)
        return None
    rng = np.random.default_rng(SEED)
    if len(train["X"]) > max_train_rows:
        idx = rng.choice(len(train["X"]), max_train_rows, replace=False)
        x_train, y_train = train["X"][idx], train["y_status"][idx]
    else:
        x_train, y_train = train["X"], train["y_status"]
    try:
        started = time.time()
        model = TabPFNClassifier(device="cpu")
        model.fit(x_train, y_train)
        result = {"model": "TabPFN", "seconds": 0.0, "test_results": {}, "note": f"local inference; trained on a {len(x_train)}-row subsample and evaluated on up to {max_eval_rows}-row subsamples due to TabPFN's local size constraints"}
        for test_name, test in test_sets.items():
            n = min(len(test["X"]), max_eval_rows)
            proba = model.predict_proba(test["X"][:n])
            result["test_results"][test_name] = classification_metrics(test["y_status"][:n], proba)
        result["seconds"] = time.time() - started
        print(f"[CLASSICAL] TabPFN: {result['seconds']:.1f}s | " + " | ".join(f"{k}_acc={v['accuracy']:.3f}" for k, v in result["test_results"].items()), flush=True)
        return result
    except Exception as exc:
        print(f"[CLASSICAL][TabPFN] failed at runtime, skipping: {exc}", flush=True)
        return {"model": "TabPFN", "error": str(exc)}


def main():
    started = time.time()
    metrics_dir().mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)

    print("[CLASSICAL] building N1 (synthetic-only) and N2 (combined) regimes ...", flush=True)
    n1 = build_n1_synthetic_only()
    n2 = build_n2_combined()

    n1_test_sets = {"synthetic_test": n1["test"]}
    n2_test_sets = {"original_real_test": n2["test_original_real"], "synthetic_holdout_test": n2["test_synthetic_holdout"]}

    runs = []
    saved_models = {}
    for name in ("CatBoost", "XGBoost", "ExtraTrees"):
        result, model = run_classifier(name, "N1_synthetic_only", n1["train"], n1["validation"], n1_test_sets)
        runs.append({"task": "classification", **result})
        saved_models[f"N1_{name}_classifier"] = model

        result, model = run_classifier(name, "N2_combined", n2["train"], n2["validation"], n2_test_sets)
        runs.append({"task": "classification", **result})
        saved_models[f"N2_{name}_classifier"] = model

        result, models = run_regressors(name, "N1_synthetic_only", n1["train"], n1["validation"], n1_test_sets)
        runs.append({"task": "regression", **result})
        saved_models[f"N1_{name}_regressors"] = models

        result, models = run_regressors(name, "N2_combined", n2["train"], n2["validation"], n2_test_sets)
        runs.append({"task": "regression", **result})
        saved_models[f"N2_{name}_regressors"] = models

    tabpfn_n1 = try_tabpfn(n1["train"], n1_test_sets)
    tabpfn_n2 = try_tabpfn(n2["train"], n2_test_sets)
    if tabpfn_n1 and "error" not in tabpfn_n1:
        runs.append({"task": "classification", "regime": "N1_synthetic_only", **tabpfn_n1})
    if tabpfn_n2 and "error" not in tabpfn_n2:
        runs.append({"task": "classification", "regime": "N2_combined", **tabpfn_n2})

    joblib.dump(saved_models, models_dir() / "classical_models.joblib")
    payload = {"runs": runs, "total_seconds": time.time() - started}
    (metrics_dir() / "classical_benchmark.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[CLASSICAL] BENCHMARK COMPLETE in {payload['total_seconds']:.1f}s -> {metrics_dir() / 'classical_benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()




