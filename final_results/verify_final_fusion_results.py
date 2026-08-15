"""Verify final fusion metrics from the committed frozen test outputs only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "frozen_fusion_test_predictions.csv"
METRICS = ROOT / "final_fusion_metrics.json"
CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
TARGETS = ["Depression", "Anxiety", "Stress"]
EXPECTED_CLASSIFICATION = {
    "accuracy": 0.6633333333333333,
    "f1_macro": 0.5223795923918323,
    "f1_weighted": 0.6538571363810042,
    "roc_auc_ovr_weighted": 0.8303076205374033,
}
EXPECTED_CONFUSION_MATRIX = [
    [189, 39, 17, 0],
    [49, 118, 18, 0],
    [26, 35, 90, 0],
    [4, 10, 4, 1],
]


def load_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 600:
        raise AssertionError(f"Expected 600 frozen test rows, found {len(rows)}")
    y_true = np.asarray([int(row["true_class_id"]) for row in rows])
    probabilities = np.asarray(
        [[float(row[f"prob_{name.lower()}"]) for name in CLASSES] for row in rows],
        dtype=np.float32,
    )
    true_regression = np.asarray(
        [[float(row[f"true_{name.lower()}"]) for name in TARGETS] for row in rows],
        dtype=np.float32,
    )
    predicted_regression = np.asarray(
        [[float(row[f"predicted_{name.lower()}"]) for name in TARGETS] for row in rows],
        dtype=np.float32,
    )
    return y_true, probabilities, true_regression, predicted_regression


def calculate() -> dict[str, object]:
    y_true, probabilities, true_regression, predicted_regression = load_predictions()
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    classification = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro", labels=labels)
        ),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted", labels=labels)
        ),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    regression: dict[str, dict[str, float]] = {}
    for index, target in enumerate(TARGETS):
        actual = true_regression[:, index]
        predicted = predicted_regression[:, index]
        mse = mean_squared_error(actual, predicted)
        regression[target] = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(actual, predicted)),
            "explained_variance": float(explained_variance_score(actual, predicted)),
        }
    overall = {
        metric: float(np.mean([regression[target][metric] for target in TARGETS]))
        for metric in ("mae", "mse", "rmse", "r2", "explained_variance")
    }
    return {
        "test_sample_count": len(y_true),
        "classification": classification,
        "confusion_matrix": matrix,
        "regression": regression,
        "regression_overall_mean": overall,
    }


def verify(results: dict[str, object]) -> None:
    classification = results["classification"]
    failures = {
        key: {"expected": expected, "actual": classification[key]}
        for key, expected in EXPECTED_CLASSIFICATION.items()
        if abs(classification[key] - expected) > 1e-12
    }
    if results["confusion_matrix"] != EXPECTED_CONFUSION_MATRIX:
        failures["confusion_matrix"] = {
            "expected": EXPECTED_CONFUSION_MATRIX,
            "actual": results["confusion_matrix"],
        }
    reference = json.loads(METRICS.read_text(encoding="utf-8"))
    if reference["provenance"]["test_sample_count"] != results["test_sample_count"]:
        failures["test_sample_count"] = {
            "expected": reference["provenance"]["test_sample_count"],
            "actual": results["test_sample_count"],
        }
    for section in ("classification", "regression", "regression_overall_mean"):
        expected_section = reference[section]
        actual_section = results[section]
        if section == "regression":
            comparisons = (
                (f"{target}.{metric}", expected_section[target][metric], actual_section[target][metric])
                for target in TARGETS
                for metric in expected_section[target]
            )
        else:
            comparisons = (
                (metric, expected_section[metric], actual_section[metric])
                for metric in expected_section
            )
        for metric, expected, actual in comparisons:
            if abs(actual - expected) > 1e-12:
                failures[f"{section}.{metric}"] = {"expected": expected, "actual": actual}
    if failures:
        raise AssertionError("Frozen final results did not reproduce:\n" + json.dumps(failures, indent=2))


if __name__ == "__main__":
    calculated = calculate()
    verify(calculated)
    print(json.dumps(calculated, indent=2))
    print("\nVERIFIED: frozen final fusion test metrics reproduce exactly.")
