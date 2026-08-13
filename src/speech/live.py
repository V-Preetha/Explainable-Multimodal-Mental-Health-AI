"""Per-epoch console reporting for the speech branch.

Reuses the generic (pipeline-agnostic) GPU-status / clock helpers already in
``src/live.py`` -- read-only introspection shared safely across branches --
but defines its own epoch printer so every field the speech benchmark spec
requires is always shown.
"""
from __future__ import annotations

import time

from src.live import clock, gpu_status  # generic, read-only helpers shared across branches


def phase(tag: str, name: str, step: str) -> None:
    print("\n" + "=" * 92, flush=True)
    print(f"[SPEECH][{tag}][{name}] {step}", flush=True)
    print(gpu_status(), flush=True)
    print("=" * 92, flush=True)


def epoch_report(
    tag: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    train_accuracy: float,
    val_accuracy: float,
    train_macro_f1: float,
    val_macro_f1: float,
    train_weighted_f1: float,
    val_weighted_f1: float,
    per_class_f1: dict[str, float],
    learning_rate: float,
    best_macro_f1: float,
    best_epoch: int,
    started: float,
) -> None:
    elapsed = time.time() - started
    eta = elapsed / max(epoch, 1) * max(total_epochs - epoch, 0)
    per_class_line = " | ".join(f"{name}={value:.3f}" for name, value in per_class_f1.items())
    print(
        f"\n[SPEECH][{tag}] Epoch {epoch:02d}/{total_epochs:02d}\n"
        f"  Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}\n"
        f"  Train Acc:  {100*train_accuracy:.2f}%   Val Acc: {100*val_accuracy:.2f}%\n"
        f"  Train Macro F1: {train_macro_f1:.4f}   Val Macro F1: {val_macro_f1:.4f}\n"
        f"  Train Weighted F1: {train_weighted_f1:.4f}   Val Weighted F1: {val_weighted_f1:.4f}\n"
        f"  Per-class Val F1: {per_class_line}\n"
        f"  LR: {learning_rate:.3e}\n"
        f"  Best Val Macro F1: {best_macro_f1:.4f} @ epoch {best_epoch:02d}\n"
        f"  {gpu_status()}\n"
        f"  Elapsed: {clock(elapsed)} | ETA: {clock(eta)}",
        flush=True,
    )

