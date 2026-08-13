from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from tqdm.auto import tqdm

from src.common import SEED, root
from src.live import clock, gpu_status, phase


R = root()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
OUT = R / "outputs" / "face"
CHECKPOINT = OUT / "best_convnext_tiny_real_only.pt"


class FERDataset(Dataset):
    """Same frozen split and preprocessing used by the prior clean ConvNeXt run."""

    def __init__(self, rows, image_size: int, train: bool):
        self.rows = rows
        geometry = (
            [
                transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandAugment(num_ops=2, magnitude=7),
            ]
            if train
            else [transforms.Resize((image_size, image_size))]
        )
        self.transform = transforms.Compose(
            geometry
            + [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, target = self.rows[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, target, str(path)


class ConvNeXtEmotion(nn.Module):
    def __init__(self, classes: int = 7, embedding_dim: int = 256):
        super().__init__()
        # Fresh FER run, initialized from the same ImageNet-1K weights as the validated baseline.
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(0.30)
        )
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(self, x):
        return self.classifier(self.embedding(self.backbone(x)))


def make_loaders(image_size: int, batch_size: int, workers: int):
    split_path = R / "configs" / "face_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    rows = {name: [(Path(path), int(label)) for path, label in values] for name, values in split.items()}
    counts = Counter(label for _, label in rows["train"])
    sample_weights = [1.0 / math.sqrt(counts[label]) for _, label in rows["train"]]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(SEED)
    )
    loaders = {}
    for name in ("train", "validation", "test"):
        loaders[name] = DataLoader(
            FERDataset(rows[name], image_size, name == "train"),
            batch_size=batch_size,
            sampler=sampler if name == "train" else None,
            shuffle=False,
            num_workers=workers,
            pin_memory=DEVICE.type == "cuda",
            persistent_workers=workers > 0,
        )
    return split_path, rows, loaders, counts


def metrics_from_arrays(y, probabilities, loss):
    pred = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(CLASSES)), zero_division=0
    )
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predicted_counts": np.bincount(pred, minlength=len(CLASSES)),
        "predictions": pred,
    }


def run_epoch(model, loader, criterion, optimizer, scaler, epoch, total_epochs):
    model.train()
    total_loss, seen, targets, probabilities = 0.0, 0, [], []
    bar = tqdm(loader, desc=f"[FACE][ConvNeXt-Tiny][REAL-ONLY] train {epoch:02d}/{total_epochs:02d}", dynamic_ncols=True)
    for step, (images, labels, _) in enumerate(bar, 1):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        batch = len(labels)
        total_loss += loss.item() * batch
        seen += batch
        targets.extend(labels.detach().cpu().tolist())
        probabilities.append(logits.detach().float().softmax(1).cpu().numpy())
        bar.set_postfix(loss=f"{total_loss / seen:.4f}", lr=f"{optimizer.param_groups[-1]['lr']:.2e}")
        if step % 80 == 0:
            tqdm.write(gpu_status())
    y, p = np.asarray(targets), np.concatenate(probabilities)
    return metrics_from_arrays(y, p, total_loss / seen)


@torch.inference_mode()
def evaluate(model, loader, criterion, description):
    model.eval()
    total_loss, seen, targets, probabilities, paths = 0.0, 0, [], [], []
    for images, labels, batch_paths in tqdm(loader, desc=description, dynamic_ncols=True, leave=False):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        batch = len(labels)
        total_loss += loss.item() * batch
        seen += batch
        targets.extend(labels.cpu().tolist())
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        paths.extend(batch_paths)
    y, p = np.asarray(targets), np.concatenate(probabilities)
    return metrics_from_arrays(y, p, total_loss / seen), y, p, paths


def write_history(history, per_class_history):
    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    pd.DataFrame(per_class_history).to_csv(OUT / "validation_per_class_metrics.csv", index=False)


def print_epoch(epoch, total, train, val, best, best_epoch, lr, epoch_seconds, elapsed, patience, eta):
    print("\n" + "=" * 96)
    print("[FACE][ConvNeXt-Tiny][REAL-ONLY]")
    print(f"Epoch {epoch:02d}/{total:02d}")
    print(f"Train Loss:          {train['loss']:.4f}")
    print(f"Train Accuracy:      {100 * train['accuracy']:.2f}%")
    print(f"Train Macro F1:      {train['macro_f1']:.4f}")
    print(f"Val Loss:            {val['loss']:.4f}")
    print(f"Val Accuracy:        {100 * val['accuracy']:.2f}%")
    print(f"Val Macro F1:        {val['macro_f1']:.4f}")
    print(f"Val Weighted F1:     {val['weighted_f1']:.4f}")
    print(f"Best Val Macro F1:   {best:.4f}")
    print(f"Best Epoch:          {best_epoch:02d}")
    print(f"Learning Rate:       {lr:.3e}")
    print(gpu_status())
    print(f"Epoch Duration:      {clock(epoch_seconds)}")
    print(f"Total Elapsed:       {clock(elapsed)}")
    print(f"ETA:                 {clock(eta)}")
    print(f"Early-stop counter:  {patience}/10")
    print(f"Best checkpoint:     {CHECKPOINT}")
    print("\nPer-class validation metrics")
    print(f"{'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Predicted':>10s}")
    for i, name in enumerate(CLASSES):
        print(f"{name:12s} {val['precision'][i]:10.4f} {val['recall'][i]:10.4f} {val['f1'][i]:10.4f} {val['predicted_counts'][i]:10d}")
    print("=" * 96, flush=True)


def plot_confusion(cm, normalized, path):
    values = cm.astype(float)
    if normalized:
        values = np.divide(values, values.sum(1, keepdims=True), out=np.zeros_like(values), where=values.sum(1, keepdims=True) != 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(7):
        for j in range(7):
            text = f"{values[i, j]:.2f}" if normalized else str(int(values[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if values[i, j] > values.max() * .55 else "black")
    ax.set_title("Normalized Confusion Matrix" if normalized else "Confusion Matrix (Counts)")
    fig.colorbar(image, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_training(history, per_class, best_epoch):
    h = pd.DataFrame(history)
    specs = [
        ("loss_curves.png", "Loss", "train_loss", "val_loss"),
        ("accuracy_curves.png", "Accuracy", "train_accuracy", "val_accuracy"),
        ("macro_f1_curves.png", "Macro F1", "train_macro_f1", "val_macro_f1"),
    ]
    for filename, title, train_col, val_col in specs:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(h.epoch, h[train_col], label="Train"); ax.plot(h.epoch, h[val_col], label="Validation")
        ax.axvline(best_epoch, color="black", ls="--", alpha=.6, label=f"Best epoch {best_epoch}"); ax.set(xlabel="Epoch", ylabel=title, title=f"Train vs Validation {title}"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(h.epoch, h.learning_rate); ax.axvline(best_epoch, color="black", ls="--", alpha=.6); ax.set(xlabel="Epoch", ylabel="Learning rate", title="Learning Rate vs Epoch"); ax.set_yscale("log"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "learning_rate_curve.png", dpi=180); plt.close(fig)
    pc = pd.DataFrame(per_class)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in CLASSES:
        group = pc[pc["class"] == name]; ax.plot(group.epoch, group.f1, label=name)
    ax.axvline(best_epoch, color="black", ls="--", alpha=.6); ax.set(xlabel="Epoch", ylabel="Validation F1", title="Per-class Validation F1"); ax.grid(alpha=.25); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(OUT / "per_class_validation_f1.png", dpi=180); plt.close(fig)


def plot_roc_pr(y, p):
    binary = label_binarize(y, classes=np.arange(7))
    aucs, aps = {}, {}
    fig_roc, ax_roc = plt.subplots(figsize=(9, 7)); fig_pr, ax_pr = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(binary[:, i], p[:, i]); aucs[name] = float(auc(fpr, tpr)); ax_roc.plot(fpr, tpr, label=f"{name} ({aucs[name]:.3f})")
        precision, recall, _ = precision_recall_curve(binary[:, i], p[:, i]); aps[name] = float(average_precision_score(binary[:, i], p[:, i])); ax_pr.plot(recall, precision, label=f"{name} (AP {aps[name]:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=.5); ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="One-vs-Rest ROC Curves"); ax_roc.grid(alpha=.25); ax_roc.legend(); fig_roc.tight_layout(); fig_roc.savefig(OUT / "roc_curves.png", dpi=180); plt.close(fig_roc)
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="One-vs-Rest Precision-Recall Curves"); ax_pr.grid(alpha=.25); ax_pr.legend(); fig_pr.tight_layout(); fig_pr.savefig(OUT / "precision_recall_curves.png", dpi=180); plt.close(fig_pr)
    return aucs, aps


def prediction_grid(indices, paths, y, pred, confidence, output, title, limit=20):
    indices = list(indices)[:limit]
    if not indices:
        return
    cols, rows = 5, math.ceil(len(indices) / 5)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.4 * rows)); axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis("off")
    for ax, index in zip(axes, indices):
        with Image.open(paths[index]) as im: ax.imshow(im.convert("L"), cmap="gray")
        ax.set_title(f"T: {CLASSES[y[index]]}\nP: {CLASSES[pred[index]]} ({confidence[index]:.3f})", fontsize=9); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def gradcam(model, image_tensor, class_index):
    activations, gradients = {}, {}
    def hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))
    handle = model.backbone.features[-1].register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model(image_tensor)
        logits[0, class_index].backward()
    handle.remove()
    feature, grad = activations["value"], gradients["value"]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam -= cam.min(); cam /= cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy()


def gradcam_grid(model, indices, dataset, paths, y, pred, output, title):
    indices = list(indices)
    if not indices:
        return
    fig, axes = plt.subplots(math.ceil(len(indices) / 4), 4, figsize=(14, 3.5 * math.ceil(len(indices) / 4))); axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis("off")
    model.eval()
    for ax, index in zip(axes, indices):
        tensor, _, _ = dataset[index]; cam = gradcam(model, tensor.unsqueeze(0).to(DEVICE), int(pred[index]))
        with Image.open(paths[index]) as im: gray = np.asarray(im.convert("L").resize((cam.shape[1], cam.shape[0])))
        ax.imshow(gray, cmap="gray"); ax.imshow(cam, cmap="jet", alpha=.42, vmin=0, vmax=1); ax.set_title(f"T: {CLASSES[y[index]]} | P: {CLASSES[pred[index]]}", fontsize=9); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def select_one_per_class(y, pred, correct):
    chosen = []
    for class_id in range(7):
        candidates = np.flatnonzero((y == class_id) & ((pred == y) if correct else (pred != y)))
        if len(candidates): chosen.append(int(candidates[0]))
    return chosen


def final_analysis(model, loaders, rows, criterion, checkpoint_payload, history, per_class_history, stopped_early):
    phase("FACE", "ConvNeXt-Tiny REAL-ONLY", "BEST-CHECKPOINT VALIDATION EVALUATION")
    validation, val_y, val_p, val_paths = evaluate(
        model, loaders["validation"], criterion, "[FACE] best-checkpoint validation batches"
    )
    val_pred = validation["predictions"]
    val_records = []
    for i, path in enumerate(val_paths):
        record = {
            "filename": path,
            "true_label": CLASSES[val_y[i]],
            "predicted_label": CLASSES[val_pred[i]],
            "confidence": float(val_p[i].max()),
            "correct": bool(val_y[i] == val_pred[i]),
        }
        record.update({f"prob_{name.lower()}": float(val_p[i, j]) for j, name in enumerate(CLASSES)})
        val_records.append(record)
    pd.DataFrame(val_records).to_csv(OUT / "validation_predictions_best_checkpoint.csv", index=False)
    validation_summary = {
        "loss": validation["loss"],
        "accuracy": validation["accuracy"],
        "macro_f1": validation["macro_f1"],
        "weighted_f1": validation["weighted_f1"],
        "per_class": {
            name: {
                "precision": float(validation["precision"][i]),
                "recall": float(validation["recall"][i]),
                "f1": float(validation["f1"][i]),
                "support": int(validation["support"][i]),
                "predicted_count": int(validation["predicted_counts"][i]),
            }
            for i, name in enumerate(CLASSES)
        },
    }
    (OUT / "validation_metrics_best_checkpoint.json").write_text(
        json.dumps(validation_summary, indent=2), encoding="utf-8"
    )
    print(
        f"Best-checkpoint validation: loss={validation['loss']:.4f} | "
        f"accuracy={100 * validation['accuracy']:.2f}% | Macro F1={validation['macro_f1']:.4f} | "
        f"Weighted F1={validation['weighted_f1']:.4f}",
        flush=True,
    )
    phase("FACE", "ConvNeXt-Tiny REAL-ONLY", "UNTOUCHED TEST EVALUATION (one pass)")
    test, y, p, paths = evaluate(model, loaders["test"], criterion, "[FACE] held-out test batches")
    pred, confidence = test["predictions"], p.max(1)
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    weighted_precision, weighted_recall, _, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    macro_auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    weighted_auc = float(roc_auc_score(y, p, multi_class="ovr", average="weighted"))
    cm = confusion_matrix(y, pred, labels=np.arange(7))
    plot_confusion(cm, False, OUT / "confusion_matrix_counts.png"); plot_confusion(cm, True, OUT / "confusion_matrix_normalized.png")
    plot_training(history, per_class_history, checkpoint_payload["epoch"])
    class_aucs, class_aps = plot_roc_pr(y, p)

    records = []
    for i, path in enumerate(paths):
        record = {"filename": path, "true_label": CLASSES[y[i]], "predicted_label": CLASSES[pred[i]], "confidence": confidence[i]}
        record.update({f"prob_{name.lower()}": p[i, j] for j, name in enumerate(CLASSES)}); record["correct"] = bool(y[i] == pred[i]); records.append(record)
    pd.DataFrame(records).to_csv(OUT / "test_predictions.csv", index=False)
    wrong = np.flatnonzero(y != pred); wrong_sorted = wrong[np.argsort(-confidence[wrong])]
    prediction_grid(wrong_sorted, paths, y, pred, confidence, OUT / "highest_confidence_incorrect.png", "20 Highest-confidence Incorrect Predictions")
    correct = np.flatnonzero(y == pred); representatives = []
    for class_id in range(7):
        candidates = correct[y[correct] == class_id]; representatives.extend(candidates[np.argsort(-confidence[candidates])[:3]].tolist())
    prediction_grid(representatives, paths, y, pred, confidence, OUT / "representative_correct.png", "Representative Correct Predictions", 21)

    confusion_rows = []
    for true_id in range(7):
        for pred_id in range(7):
            if true_id != pred_id and cm[true_id, pred_id]: confusion_rows.append({"true_label": CLASSES[true_id], "predicted_label": CLASSES[pred_id], "count": int(cm[true_id, pred_id])})
    confusion_rows.sort(key=lambda x: x["count"], reverse=True); pd.DataFrame(confusion_rows).to_csv(OUT / "frequent_confusions.csv", index=False)

    phase("FACE", "ConvNeXt-Tiny REAL-ONLY", "GRAD-CAM")
    test_dataset = loaders["test"].dataset
    gradcam_grid(model, select_one_per_class(y, pred, True), test_dataset, paths, y, pred, OUT / "gradcam_correct.png", "Grad-CAM: Correct Predictions (one per available class)")
    gradcam_grid(model, select_one_per_class(y, pred, False), test_dataset, paths, y, pred, OUT / "gradcam_misclassified.png", "Grad-CAM: Misclassified Predictions (one per available true class)")

    best_row = next(row for row in history if row["epoch"] == checkpoint_payload["epoch"])
    acc_gap = best_row["train_accuracy"] - best_row["val_accuracy"]
    f1_gap = best_row["train_macro_f1"] - best_row["val_macro_f1"]
    after = [row["val_macro_f1"] for row in history if row["epoch"] > checkpoint_payload["epoch"]]
    post_best_drop = checkpoint_payload["val_macro_f1"] - min(after) if after else 0.0
    if best_row["train_accuracy"] < .70 and best_row["val_accuracy"] < .65: diagnosis = "UNDERFITTING"
    elif acc_gap > .20 or f1_gap > .20 or post_best_drop > .10: diagnosis = "SEVERE OVERFITTING"
    elif acc_gap > .10 or f1_gap > .10 or post_best_drop > .04: diagnosis = "MILD OVERFITTING"
    else: diagnosis = "HEALTHY FIT"

    per_class = {}
    for i, name in enumerate(CLASSES):
        per_class[name] = {"precision": float(test["precision"][i]), "recall": float(test["recall"][i]), "f1": float(test["f1"][i]), "support": int(test["support"][i]), "roc_auc": class_aucs[name], "average_precision": class_aps[name]}
    weakest = min(CLASSES, key=lambda name: per_class[name]["f1"])
    result = {
        "architecture": "torchvision ConvNeXt-Tiny; fresh FER head; ImageNet-1K V1 initialization",
        "data": "original real FER2013 only for train/validation/test; frozen face_deep_split.json",
        "selection": "highest validation Macro F1 checkpoint; test evaluated once after selection",
        "epochs_completed": len(history), "stopped_early": stopped_early, "best_epoch": checkpoint_payload["epoch"],
        "best_validation_accuracy": checkpoint_payload["val_accuracy"], "best_validation_macro_f1": checkpoint_payload["val_macro_f1"], "best_validation_weighted_f1": checkpoint_payload["val_weighted_f1"],
        "best_checkpoint_validation_evaluation": validation_summary,
        "test_loss": test["loss"], "test_accuracy": test["accuracy"], "test_macro_precision": float(macro_precision), "test_macro_recall": float(macro_recall), "test_macro_f1": test["macro_f1"],
        "test_weighted_precision": float(weighted_precision), "test_weighted_recall": float(weighted_recall), "test_weighted_f1": test["weighted_f1"], "test_macro_roc_auc_ovr": macro_auc, "test_weighted_roc_auc_ovr": weighted_auc,
        "per_class": per_class, "weakest_class": weakest, "most_confused_pair": confusion_rows[0] if confusion_rows else None,
        "diagnosis": diagnosis, "diagnosis_evidence": {"accuracy_gap_at_best": acc_gap, "macro_f1_gap_at_best": f1_gap, "maximum_post_best_val_macro_f1_drop": post_best_drop},
        "checkpoint": str(CHECKPOINT), "metrics_directory": str(OUT), "plots_directory": str(OUT), "split_sizes": {k: len(v) for k, v in rows.items()},
    }
    (OUT / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([{ "class": name, **values } for name, values in per_class.items()]).to_csv(OUT / "test_per_class_metrics.csv", index=False)
    print("\n" + "=" * 96); print("CONVNEXT-TINY REAL-ONLY BASELINE")
    print(f"Epochs completed: {len(history)}\nStopped early: {'YES' if stopped_early else 'NO'}\nBest epoch: {checkpoint_payload['epoch']}")
    print(f"\nBest Val Accuracy: {100*checkpoint_payload['val_accuracy']:.2f}%\nBest Val Macro F1: {checkpoint_payload['val_macro_f1']:.4f}\nBest Val Weighted F1: {checkpoint_payload['val_weighted_f1']:.4f}")
    print(f"\nTest Loss: {test['loss']:.4f}\nTest Accuracy: {100*test['accuracy']:.2f}%\nTest Macro Precision: {macro_precision:.4f}\nTest Macro Recall: {macro_recall:.4f}\nTest Macro F1: {test['macro_f1']:.4f}")
    print(f"Test Weighted Precision: {weighted_precision:.4f}\nTest Weighted Recall: {weighted_recall:.4f}\nTest Weighted F1: {test['weighted_f1']:.4f}\nTest Macro ROC-AUC: {macro_auc:.4f}\nTest Weighted ROC-AUC: {weighted_auc:.4f}")
    print("\nPer-class F1:"); [print(f"{name}: {per_class[name]['f1']:.4f}") for name in CLASSES]
    pair = confusion_rows[0] if confusion_rows else {"true_label": "N/A", "predicted_label": "N/A", "count": 0}
    print(f"\nWeakest class: {weakest}\nMost confused class pair: {pair['true_label']} -> {pair['predicted_label']}: {pair['count']}")
    print(f"Training diagnosis: {diagnosis}\nBest checkpoint: {CHECKPOINT}\nMetrics directory: {OUT}\nPlots directory: {OUT}")
    print("=" * 96, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=.001)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.max_epochs != 60: print(f"NOTICE: scheduler horizon follows requested max_epochs={args.max_epochs}")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    phase("FACE", "ConvNeXt-Tiny REAL-ONLY", "FROZEN SPLIT AUDIT")
    split_path, rows, loaders, counts = make_loaders(args.image_size, args.batch_size, args.workers)
    for name, values in rows.items():
        for path, _ in tqdm(values, desc=f"Auditing {name} real-image manifest", dynamic_ncols=True):
            if not path.is_file(): raise FileNotFoundError(path)
    print(f"Split source (not regenerated): {split_path}\nSplit sizes: { {k: len(v) for k,v in rows.items()} }\nReal train class counts: { {CLASSES[k]: v for k,v in sorted(counts.items())} }\nSynthetic samples: 0", flush=True)

    if args.evaluate_only:
        phase("FACE", "ConvNeXt-Tiny REAL-ONLY", "EVALUATION ONLY - NO TRAINING")
        payload = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        if int(payload["epoch"]) != 22:
            raise RuntimeError(f"Expected selected epoch 22 checkpoint, found epoch {payload['epoch']}")
        model = ConvNeXtEmotion().to(DEVICE)
        model.load_state_dict(payload["state_dict"])
        criterion = nn.CrossEntropyLoss(label_smoothing=.08)
        history = pd.read_csv(OUT / "training_history.csv").to_dict("records")
        per_class_history = pd.read_csv(OUT / "validation_per_class_metrics.csv").to_dict("records")
        print(
            f"Selected preserved checkpoint: epoch {payload['epoch']} | "
            f"Val Macro F1 {payload['val_macro_f1']:.7f} | {CHECKPOINT}",
            flush=True,
        )
        final_analysis(model, loaders, rows, criterion, payload, history, per_class_history, True)
        return

    phase("FACE", "ConvNeXt-Tiny REAL-ONLY", "FRESH 60-EPOCH FINE-TUNING")
    print("Initialization: fresh FER classifier run from ImageNet-1K V1 weights; NO prior FER checkpoint loaded.")
    model = ConvNeXtEmotion().to(DEVICE)
    for parameter in model.backbone.parameters(): parameter.requires_grad = False
    criterion = nn.CrossEntropyLoss(label_smoothing=.08)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": list(model.embedding.parameters()) + list(model.classifier.parameters()), "lr": 3e-4},
    ], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=2e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    history, per_class_history = [], []
    best_raw, patience_reference, best_epoch, stale = -1.0, -1.0, 0, 0
    started = time.time()
    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.time()
        if epoch == args.warmup_epochs + 1:
            for parameter in model.backbone.parameters(): parameter.requires_grad = True
            print("[FACE][ConvNeXt-Tiny][REAL-ONLY] Backbone unfrozen for discriminative fine-tuning.", flush=True)
        lr = optimizer.param_groups[-1]["lr"]
        train = run_epoch(model, loaders["train"], criterion, optimizer, scaler, epoch, args.max_epochs)
        val, _, _, _ = evaluate(model, loaders["validation"], criterion, f"[FACE] validation {epoch:02d}/{args.max_epochs:02d}")
        if val["macro_f1"] > best_raw:
            best_raw, best_epoch = val["macro_f1"], epoch
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "epoch": epoch, "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"], "image_size": args.image_size, "architecture": "torchvision ConvNeXt-Tiny ImageNet1K V1"}, CHECKPOINT)
        if val["macro_f1"] > patience_reference + args.min_delta:
            patience_reference, stale = val["macro_f1"], 0
        else: stale += 1
        epoch_seconds, elapsed = time.time() - epoch_started, time.time() - started
        eta = elapsed / epoch * (args.max_epochs - epoch)
        history.append({"epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"], "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"], "train_macro_f1": train["macro_f1"], "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"], "learning_rate": lr, "epoch_seconds": epoch_seconds})
        for i, name in enumerate(CLASSES): per_class_history.append({"epoch": epoch, "class": name, "precision": val["precision"][i], "recall": val["recall"][i], "f1": val["f1"][i], "support": val["support"][i], "predicted_count": val["predicted_counts"][i]})
        write_history(history, per_class_history)
        print_epoch(epoch, args.max_epochs, train, val, best_raw, best_epoch, lr, epoch_seconds, elapsed, stale, eta)
        scheduler.step()
        if stale >= args.patience:
            print(f"EARLY STOPPING: no >= {args.min_delta:.4f} validation Macro F1 improvement for {args.patience} epochs.", flush=True); break
    stopped_early = len(history) < args.max_epochs
    phase("FACE", "ConvNeXt-Tiny REAL-ONLY", "RELOAD BEST VALIDATION MACRO F1 CHECKPOINT")
    payload = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False); model.load_state_dict(payload["state_dict"])
    print(f"Reloaded epoch {payload['epoch']} with validation Macro F1 {payload['val_macro_f1']:.4f}: {CHECKPOINT}", flush=True)
    final_analysis(model, loaders, rows, criterion, payload, history, per_class_history, stopped_early)


if __name__ == "__main__":
    main()

