"""train implementation for the curated submission repository."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.speech.common import (
    CLASSES, NUM_CLASSES, SEED, TEST_ACTORS, TRAIN_ACTORS, VAL_ACTORS,
    assert_zero_actor_overlap, load_ravdess_rows, root, split_indices,
)
from src.speech.dataset import FrameDataset, collate_frames
from src.speech.features import FRAME_DIM, MODEL_ID, extract_real_frame_embeddings
from src.speech.live import epoch_report, phase
from src.speech.metrics_utils import (
    compute_metrics, per_class_f1_dict, save_confusion_matrices,
    save_history_csv, save_per_class_table, save_pr_curves,
    save_prediction_probabilities, save_roc_curves,
)
from src.speech.models import SpeechEmotionMLP

R = root()
OUT_DIR = R / "artifacts" / "speech" / "strict"


def build_loaders(payload: dict, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    records = payload["records"]
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]
    train_loader = DataLoader(FrameDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=collate_frames, num_workers=0)
    val_loader = DataLoader(FrameDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=collate_frames, num_workers=0)
    test_loader = DataLoader(FrameDataset(test_records), batch_size=batch_size, shuffle=False, collate_fn=collate_frames, num_workers=0)
    return train_loader, val_loader, test_loader


@torch.inference_mode()
def run_inference(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    labels, probabilities, embeddings, paths, actors_all = [], [], [], [], []
    total_loss = 0.0
    for frames, mask, y, path_batch, actor_batch in loader:
        frames, mask, y = frames.to(device), mask.to(device), y.to(device)
        embedding, logits, _ = model(frames, mask)
        total_loss += criterion(logits, y).item() * len(y)
        labels.extend(y.cpu().tolist())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
        embeddings.append(embedding.cpu())
        paths.extend(path_batch)
        actors_all.extend(actor_batch.tolist())
    y_true = np.asarray(labels)
    y_prob = np.concatenate(probabilities)
    loss = total_loss / len(loader.dataset)
    return y_true, y_prob, torch.cat(embeddings), paths, actors_all, loss


def export_embeddings(model: nn.Module, device: str, payload: dict, indices: dict[str, np.ndarray], out_path: Path) -> None:
    model.eval()
    export = {}
    records = payload["records"]
    with torch.inference_mode():
        for split_name, idx in indices.items():
            split_records = [records[i] for i in idx]
            loader = DataLoader(FrameDataset(split_records), batch_size=32, shuffle=False, collate_fn=collate_frames, num_workers=0)
            zs, ys, paths, actors = [], [], [], []
            for frames, mask, y, path_batch, actor_batch in loader:
                embedding, _, _ = model(frames.to(device), mask.to(device))
                zs.append(embedding.cpu())
                ys.append(y)
                paths.extend(path_batch)
                actors.append(actor_batch)
            export[split_name] = {"embeddings": torch.cat(zs), "labels": torch.cat(ys), "paths": paths, "actors": torch.cat(actors)}
    torch.save({"embedding_dim": export["train"]["embeddings"].shape[1], "model_id": MODEL_ID, "pooling": "attentive statistics", "splits": export}, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=35, help="30-40 max")
    parser.add_argument("--patience", type=int, default=7, help="early stopping patience, 6-8")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    assert 30 <= args.max_epochs <= 40, "spec requires 30-40 max epochs"
    assert 6 <= args.patience <= 8, "spec requires early stopping patience 6-8"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase("STRICT", "RAVDESS", "load, deduplicate, verify actor-independent split")
    rows = load_ravdess_rows()
    assert_zero_actor_overlap()
    train_idx, val_idx, test_idx = split_indices(rows)
    print(f"[STRICT] split sizes -> train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    print(f"[STRICT] train actors={TRAIN_ACTORS}", flush=True)
    print(f"[STRICT] val actors={VAL_ACTORS}", flush=True)
    print(f"[STRICT] test actors={TEST_ACTORS}", flush=True)
    train_actor_set = {rows[i]["actor"] for i in train_idx}
    val_actor_set = {rows[i]["actor"] for i in val_idx}
    test_actor_set = {rows[i]["actor"] for i in test_idx}
    assert not (train_actor_set & val_actor_set or train_actor_set & test_actor_set or val_actor_set & test_actor_set)
    print("[STRICT] verified zero actor overlap across train/val/test", flush=True)

    phase("STRICT", MODEL_ID, "emotion2vec+ frame embedding extraction (real audio, conservative trim only)")
    payload = extract_real_frame_embeddings(rows, device)

    train_loader, val_loader, test_loader = build_loaders(payload, train_idx, val_idx, test_idx, args.batch_size)

    model = SpeechEmotionMLP(input_dim=FRAME_DIM, embedding_dim=256, classes=NUM_CLASSES).to(device)
    train_labels = np.array([CLASSES.index(rows[i]["emotion"]) for i in train_idx])
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = torch.tensor(len(train_labels) / (NUM_CLASSES * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.02)

    checkpoint_path = OUT_DIR / "best_checkpoint.pt"
    best_f1, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.time()

    phase("STRICT", "SpeechEmotionMLP", f"training up to {args.max_epochs} epochs, early stop patience {args.patience}")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_loss = 0.0
        train_labels_epoch, train_probs_epoch = [], []
        for frames, mask, y, _paths, _actors in train_loader:
            frames, mask, y = frames.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits, _ = model(frames, mask)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item() * len(y)
            train_labels_epoch.extend(y.detach().cpu().tolist())
            train_probs_epoch.append(logits.detach().softmax(dim=1).cpu().numpy())
        train_loss = running_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(np.asarray(train_labels_epoch), np.concatenate(train_probs_epoch), CLASSES)

        val_y, val_prob, _, _, _, val_loss = run_inference(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_y, val_prob, CLASSES)
        val_per_class_f1 = per_class_f1_dict(val_y, val_prob, CLASSES)

        scheduler.step()
        improved = val_metrics["f1_macro"] > best_f1
        if improved:
            best_f1, best_epoch, stale = val_metrics["f1_macro"], epoch, 0
            torch.save({
                "state_dict": model.state_dict(), "classes": CLASSES, "input_dim": FRAME_DIM,
                "embedding_dim": 256, "model_id": MODEL_ID, "pooling": "attentive statistics",
                "validation_macro_f1": best_f1, "epoch": epoch,
            }, checkpoint_path)
        else:
            stale += 1

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["f1_macro"], "val_macro_f1": val_metrics["f1_macro"],
            "train_weighted_f1": train_metrics["f1_weighted"], "val_weighted_f1": val_metrics["f1_weighted"],
            "val_roc_auc_macro_ovr": val_metrics["roc_auc_macro_ovr"],
            "learning_rate": optimizer.param_groups[0]["lr"], "improved": improved,
        })
        epoch_report(
            "STRICT", epoch, args.max_epochs, train_loss, val_loss,
            train_metrics["accuracy"], val_metrics["accuracy"],
            train_metrics["f1_macro"], val_metrics["f1_macro"],
            train_metrics["f1_weighted"], val_metrics["f1_weighted"],
            val_per_class_f1, optimizer.param_groups[0]["lr"], best_f1, best_epoch, started,
        )
        if stale >= args.patience:
            print(f"[STRICT] early stopping: no Val Macro F1 improvement in {stale} epochs", flush=True)
            break

    save_history_csv(history, OUT_DIR / "training_history.csv")

    phase("STRICT", "SpeechEmotionMLP", "final evaluation on held-out test actors")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    test_y, test_prob, _, test_paths, test_actors, test_loss = run_inference(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_y, test_prob, CLASSES)
    test_metrics.update({
        "loss": test_loss, "model_id": MODEL_ID, "architecture": "emotion2vec+ frozen encoder + attentive statistics pooling + MLP head",
        "train_actors": TRAIN_ACTORS, "validation_actors": VAL_ACTORS, "test_actors": TEST_ACTORS, "actor_overlap": False,
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "best_validation_macro_f1": best_f1, "best_epoch": best_epoch, "epochs_completed": len(history),
        "training_seconds": time.time() - started, "checkpoint": str(checkpoint_path),
        "real_audio_only": True, "trim": "conservative leading/trailing only; internal pauses preserved",
        "deduplication": "filename + content-hash",
    })

    save_confusion_matrices(test_y, test_prob, CLASSES, OUT_DIR, "test")
    save_roc_curves(test_y, test_prob, CLASSES, OUT_DIR, "test")
    save_pr_curves(test_y, test_prob, CLASSES, OUT_DIR, "test")
    save_per_class_table(test_metrics, CLASSES, OUT_DIR, "test")
    save_prediction_probabilities(test_paths, test_y, test_prob, CLASSES, OUT_DIR, "test")

    export_embeddings(model, device, payload, {"train": train_idx, "validation": val_idx, "test": test_idx}, OUT_DIR / "speech_embeddings.pt")

    (OUT_DIR / "final_metrics.json").write_text(json.dumps(test_metrics, indent=2, default=str), encoding="utf-8")
    print("[STRICT] BENCHMARK COMPLETE", flush=True)
    print(json.dumps({
        "accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["f1_macro"],
        "weighted_f1": test_metrics["f1_weighted"], "roc_auc_macro_ovr": test_metrics["roc_auc_macro_ovr"],
        "roc_auc_weighted_ovr": test_metrics["roc_auc_weighted_ovr"], "checkpoint": str(checkpoint_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()



