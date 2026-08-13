from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MAX_SCORES = np.array([34.0, 24.0, 39.0], dtype=np.float32)
FACE_STRESS = {"Happy": 0, "Neutral": 0, "Sad": 1, "Surprise": 1, "Fear": 2, "Disgust": 2, "Angry": 3}
AUDIO_STRESS = {"neutral": 0, "calm": 0, "happy": 0, "sad": 1, "surprised": 1, "fearful": 2, "angry": 2, "disgust": 3}


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    value = os.environ.get("AEGIS_DATA_ROOT")
    if not value:
        raise RuntimeError("Set AEGIS_DATA_ROOT to the directory containing the local datasets.")
    return Path(value).expanduser().resolve()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

