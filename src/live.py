from __future__ import annotations

import subprocess
import time

import torch


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds)); hours, seconds = divmod(seconds, 3600); minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def gpu_status() -> str:
    if not torch.cuda.is_available():
        return "GPU unavailable (CPU mode)"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=False)
        util, used, total = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        return f"GPU {util}% | VRAM {float(used)/1024:.2f}/{float(total)/1024:.2f} GB"
    except Exception:
        return f"CUDA allocated {torch.cuda.memory_allocated()/2**30:.2f} GB"


def phase(branch: str, model: str, name: str) -> None:
    print(f"\n[{branch}][{model}] {name}\n{gpu_status()}", flush=True)


def epoch_report(branch, model, epoch, total_epochs, train_loss, val_loss, val_accuracy, val_macro_f1, best_f1, best_epoch, learning_rate, started):
    elapsed = time.time() - started
    eta = elapsed / max(epoch, 1) * max(total_epochs - epoch, 0)
    print(f"[{branch}][{model}] epoch {epoch:02d}/{total_epochs:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f} best={best_f1:.4f}@{best_epoch} lr={learning_rate:.3e} elapsed={clock(elapsed)} eta={clock(eta)}", flush=True)


def ensure_run_dirs(project_root):
    for relative in ("outputs", "models", "results"):
        (project_root / relative).mkdir(parents=True, exist_ok=True)

