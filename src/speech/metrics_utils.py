"""Metric computation, plotting, and CSV export shared by both speech experiments."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str]) -> dict:
    y_pred = y_prob.argmax(axis=1)
    n_classes = len(classes)
    report = classification_report(y_true, y_pred, labels=list(range(n_classes)), target_names=classes, output_dict=True, zero_division=0)
    try:
        roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(n_classes)))
        roc_auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=list(range(n_classes)))
    except ValueError:
        roc_auc_macro = roc_auc_weighted = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_macro_ovr": float(roc_auc_macro),
        "roc_auc_weighted_ovr": float(roc_auc_weighted),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }


def per_class_f1_dict(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str]) -> dict[str, float]:
    y_pred = y_prob.argmax(axis=1)
    scores = f1_score(y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0)
    return {name: float(score) for name, score in zip(classes, scores)}


def save_history_csv(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_confusion_matrices(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = y_prob.argmax(axis=1)
    n_classes = len(classes)
    raw = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(all="ignore"):
        normalized = raw.astype(np.float64) / raw.sum(axis=1, keepdims=True)
    normalized = np.nan_to_num(normalized)

    for name, matrix, fmt in (("raw", raw, "d"), ("normalized", normalized, ".2f")):
        csv_path = out_dir / f"{prefix}_confusion_matrix_{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + classes)
            for class_name, row in zip(classes, matrix):
                writer.writerow([class_name] + list(row))

        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(n_classes)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{prefix} confusion matrix ({name})")
        threshold = matrix.max() / 2.0 if matrix.size else 0
        for i in range(n_classes):
            for j in range(n_classes):
                value = matrix[i, j]
                text = format(value, fmt)
                ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black", fontsize=8)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)


def save_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(classes):
        if y_bin[:, index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, index], y_prob[:, index])
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={area:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{prefix} ROC curves (one-vs-rest)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_roc_curves.png", dpi=150)
    plt.close(fig)


def save_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(classes):
        if y_bin[:, index].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, index], y_prob[:, index])
        area = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AP={area:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{prefix} Precision-Recall curves")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_pr_curves.png", dpi=150)
    plt.close(fig)


def save_per_class_table(metrics: dict, classes: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_per_class_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name in classes:
            row = metrics["per_class"][name]
            writer.writerow([name, f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1-score']:.4f}", int(row["support"])])
        for key in ("macro avg", "weighted avg"):
            row = metrics["per_class"][key]
            writer.writerow([key, f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1-score']:.4f}", int(row["support"])])


def save_prediction_probabilities(paths: list[str], y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], out_dir: Path, prefix: str, id_column: str = "path") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_prediction_probabilities.csv"
    y_pred = y_prob.argmax(axis=1)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([id_column, "true_label", "predicted_label", "correct"] + [f"prob_{name}" for name in classes])
        for file_path, true_index, pred_index, probabilities in zip(paths, y_true, y_pred, y_prob):
            writer.writerow([file_path, classes[true_index], classes[pred_index], int(true_index == pred_index)] + [f"{p:.6f}" for p in probabilities])

