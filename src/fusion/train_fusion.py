"""train fusion implementation for the curated submission repository."""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.fusion.dataset import WeakAligned, load_face_cache, load_numerical_cache, load_speech_cache
from src.numerical.common import REGRESSION_TARGETS, STATUS_CLASSES, root
from src.numerical.metrics import regression_metrics
from src.common import MAX_SCORES
from src.fusion.model import GatedFusionMultiTask

R = root()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = R / "outputs" / "fusion" / "models"
METRICS_DIR = R / "outputs" / "fusion" / "metrics"
SPLITS_DIR = R / "outputs" / "fusion" / "splits"
CONFIGS = [
    ("numerical_only", (0, 0, 1), "probabilistic"),
    ("image_numerical", (1, 0, 1), "probabilistic"),
    ("audio_numerical", (0, 1, 1), "probabilistic"),
    ("image_audio", (1, 1, 0), "probabilistic"),
    ("all_modalities", (1, 1, 1), "probabilistic"),
    ("same_class_control", (1, 1, 1), "same"),
]


def evaluate_metrics(y: np.ndarray, p: np.ndarray, score: np.ndarray, true_score: np.ndarray) -> dict:
    pred = p.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(STATUS_CLASSES)))).tolist(),
        "per_class": classification_report(y, pred, labels=list(range(len(STATUS_CLASSES))), target_names=STATUS_CLASSES, output_dict=True, zero_division=0),
        "regression": regression_metrics(true_score, score, REGRESSION_TARGETS),
    }
    try:
        out["roc_auc_ovr_weighted"] = float(roc_auc_score(y, p, multi_class="ovr", average="weighted", labels=list(range(len(STATUS_CLASSES)))))
    except ValueError:
        out["roc_auc_ovr_weighted"] = float("nan")
    return out


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    ys, ps, ss, ts, ws = [], [], [], [], []
    for zi, za, zn, y, s, a, *_ in loader:
        logits, score, weights, _ = model(zi.to(DEVICE), za.to(DEVICE), zn.to(DEVICE), a.to(DEVICE))
        ys += y.tolist()
        ps.append(logits.softmax(dim=1).cpu().numpy())
        ss.append(score.cpu().numpy() * MAX_SCORES)
        ts.append(s.numpy() * MAX_SCORES)
        ws.append(weights.cpu().numpy())
    y = np.array(ys)
    p = np.concatenate(ps)
    score = np.concatenate(ss)
    true_score = np.concatenate(ts)
    return evaluate_metrics(y, p, score, true_score), np.concatenate(ws)


def train_config(name, modalities, alignment, face, face_classes, speech, speech_classes, numerical, max_epochs, patience):
    datasets = {s: WeakAligned(s, face, face_classes, speech, speech_classes, numerical, alignment, modalities) for s in ("train", "validation", "test")}
    loaders = {s: DataLoader(d, batch_size=128, shuffle=(s == "train"), num_workers=0) for s, d in datasets.items()}
    model = GatedFusionMultiTask().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-4)
    ce_loss = nn.CrossEntropyLoss()
    huber_loss = nn.SmoothL1Loss()
    checkpoint_path = MODEL_DIR / f"multimodal_{name}.pt"
    best_f1, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.time()
    print(f"\n[FUSION][{name}] modalities={modalities} alignment={alignment} training up to {max_epochs} epochs, patience {patience}", flush=True)
    for epoch in range(1, max_epochs + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for zi, za, zn, y, s, a, *_ in loaders["train"]:
            zi, za, zn, y, s, a = zi.to(DEVICE), za.to(DEVICE), zn.to(DEVICE), y.to(DEVICE), s.to(DEVICE), a.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits, score, _, _ = model(zi, za, zn, a, modality_dropout=0.12)
            loss = ce_loss(logits, y) + 0.3 * huber_loss(score, s)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
        val_metrics, _ = evaluate(model, loaders["validation"])
        train_loss = running_loss / len(datasets["train"])
        improved = val_metrics["f1_macro"] > best_f1
        if improved:
            best_f1, best_epoch, stale = val_metrics["f1_macro"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "name": name, "modalities": modalities, "alignment": alignment, "validation_macro_f1": best_f1}, checkpoint_path)
        else:
            stale += 1
        mean_reg_rmse = float(np.mean([v["rmse"] for v in val_metrics["regression"].values()]))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_metrics["f1_macro"], "val_weighted_f1": val_metrics["f1_weighted"], "val_mean_regression_rmse": mean_reg_rmse, "improved": improved})
        print(f"[FUSION][{name}] epoch {epoch:02d}/{max_epochs:02d} loss={train_loss:.4f} val_macro_f1={val_metrics['f1_macro']:.4f} val_mean_rmse={mean_reg_rmse:.3f} best={best_f1:.4f}@{best_epoch:02d}", flush=True)
        if stale >= patience:
            print(f"[FUSION][{name}] early stopping: no val Macro F1 improvement in {stale} epochs", flush=True)
            break
    saved = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    test_metrics, attention_weights = evaluate(model, loaders["test"])
    result = {
        "name": name, "modalities": modalities, "alignment": alignment,
        "validation_macro_f1": best_f1, "best_epoch": best_epoch, "epochs": len(history),
        "seconds": time.time() - started, "checkpoint": str(checkpoint_path),
        "test": test_metrics, "history": history,
        "mean_attention_weights": dict(zip(["face", "speech", "numerical"], attention_weights.mean(axis=0).tolist())),
    }
    return result, datasets


def write_manifests(datasets: dict, face_classes, speech_classes):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, dataset in datasets.items():
        dataset.set_epoch(0)
        out_name = "val" if split == "validation" else split
        path = SPLITS_DIR / f"multimodal_{out_name}_manifest.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["numerical_row_id", "face_file", "speech_file", "face_emotion", "speech_emotion", "target_status", "alignment_type"])
            writer.writeheader()
            for i in range(len(dataset)):
                *_, row_id, face_idx, speech_idx, alignment_type = dataset[i]
                writer.writerow({
                    "numerical_row_id": row_id,
                    "face_file": dataset.face["paths"][face_idx],
                    "speech_file": dataset.speech["paths"][speech_idx],
                    "face_emotion": face_classes[int(dataset.face["labels"][face_idx])],
                    "speech_emotion": speech_classes[int(dataset.speech["labels"][speech_idx])],
                    "target_status": STATUS_CLASSES[int(dataset.num["status_labels"][i])],
                    "alignment_type": alignment_type,
                })


def main(max_epochs: int = 50, patience: int = 8):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    face, face_classes = load_face_cache()
    speech, speech_classes = load_speech_cache()
    numerical = load_numerical_cache()
    print(f"[FUSION] caches loaded: face_dim={face['train']['embeddings'].shape[1]} speech_dim={speech['train']['embeddings'].shape[1]} numerical_dim={numerical['train']['embeddings'].shape[1]}", flush=True)

    runs, main_datasets = [], None
    for name, modalities, alignment in CONFIGS:
        result, datasets = train_config(name, modalities, alignment, face, face_classes, speech, speech_classes, numerical, max_epochs, patience)
        runs.append(result)
        if name == "all_modalities":
            main_datasets = datasets

    selected = next(r for r in runs if r["name"] == "all_modalities")
    best_state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    torch.save(best_state, MODEL_DIR / "multimodal_fusion_best.pt")
    write_manifests(main_datasets, face_classes, speech_classes)

    report = {
        "architecture": {
            "face_encoder": "ConvNeXt-Tiny (torchvision, ImageNet1K-pretrained) embedding 256D",
            "speech_encoder": "emotion2vec+ frozen encoder + attentive statistics pooling 256D",
            "numerical_encoder": "18->64->128 grouped MLP 128D",
            "fusion": "gated modality attention (256D projections) -> shared trunk 256->128",
            "heads": "4-class status + 3-target normalized regression",
        },
        "alignment": "weakly supervised class-conditional probabilistic: 70% same stress category / 20% adjacent / 10% conflicting; never participant-paired",
        "classification_loss": "CrossEntropy",
        "regression_loss": "0.3 x SmoothL1 on MAX_SCORES-normalized targets",
        "modality_dropout": 0.12,
        "selected": selected,
        "ablations": runs,
        "scientific_warning": "Metrics evaluate the weak class-conditional construction across three independent sample spaces (face/speech/numerical are not participant-paired). Same-class control is retained only as a diagnostic upper bound, not a deployment candidate.",
    }
    (METRICS_DIR / "final_multimodal_metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("[FUSION] MULTIMODAL FUSION COMPLETE", flush=True)
    print(json.dumps({
        "selected_validation_macro_f1": selected["validation_macro_f1"],
        "test_accuracy": selected["test"]["accuracy"], "test_macro_f1": selected["test"]["f1_macro"],
        "test_weighted_f1": selected["test"]["f1_weighted"], "test_roc_auc_ovr_weighted": selected["test"]["roc_auc_ovr_weighted"],
        "mean_attention_weights": selected["mean_attention_weights"],
        "checkpoint": str(MODEL_DIR / "multimodal_fusion_best.pt"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()



