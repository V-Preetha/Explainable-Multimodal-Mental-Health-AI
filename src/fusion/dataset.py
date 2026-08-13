"""Weak class-conditional alignment across the three modality embedding
caches (face 256D, speech 256D, numerical 128D).

There is no verified participant identifier joining the face dataset
(FER2013-style), RAVDESS, and the structured numerical CSV -- they are three
independent sample spaces. Following the alignment strategy already
established and documented in this project (see ``scripts/train_fusion.py``
and the branch README's "Scientific boundary" section), the numerical rows
are treated as anchors; a face example and a speech example are drawn per
anchor, per epoch, from the matching stress category 70% of the time, an
adjacent category 20% of the time, and a conflicting category 10% of the
time. This is a constructed research task, not participant-paired
supervision, and must always be reported as such.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.numerical.common import SEED, root
from src.speech.common import CLASSES as SPEECH_CLASSES
from src.common import AUDIO_STRESS, FACE_STRESS, MAX_SCORES

R = root()
FACE_CACHE = R / "outputs" / "embeddings" / "face.pt"
SPEECH_CACHE = R / "outputs" / "embeddings" / "speech.pt"
NUMERICAL_CACHE = R / "outputs" / "embeddings" / "numerical.pt"


def load_face_cache():
    payload = torch.load(FACE_CACHE, map_location="cpu", weights_only=False)
    return {split: payload[split] for split in ("train", "validation", "test")}, payload["classes"]


def load_speech_cache():
    payload = torch.load(SPEECH_CACHE, map_location="cpu", weights_only=False)
    return payload["splits"], SPEECH_CLASSES


def load_numerical_cache():
    payload = torch.load(NUMERICAL_CACHE, map_location="cpu", weights_only=False)
    return payload["splits"]


class WeakAligned(Dataset):
    def __init__(self, split, face, face_classes, speech, speech_classes, numerical, alignment="probabilistic", modalities=(1, 1, 1)):
        self.split = split
        self.face = face[split]
        self.speech = speech[split]
        self.num = numerical[split]
        self.alignment = alignment
        self.modalities = modalities
        self.epoch = 0
        self.face_by_category: dict[int, list[int]] = {}
        self.speech_by_category: dict[int, list[int]] = {}
        for i, y in enumerate(self.face["labels"].tolist()):
            self.face_by_category.setdefault(FACE_STRESS[face_classes[y]], []).append(i)
        for i, y in enumerate(self.speech["labels"].tolist()):
            self.speech_by_category.setdefault(AUDIO_STRESS[speech_classes[y]], []).append(i)
        for category in range(4):
            if category not in self.face_by_category:
                raise RuntimeError(f"[{split}] no face examples fall into stress category {category}")
            if category not in self.speech_by_category:
                raise RuntimeError(f"[{split}] no speech examples fall into stress category {category}")

    def __len__(self) -> int:
        return len(self.num["row_ids"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def choose_category(self, status: int, rng: np.random.Generator) -> tuple[int, str]:
        if self.alignment == "same":
            return status, "same"
        u = rng.random()
        if u < 0.7:
            return status, "same"
        if u < 0.9:
            adjacent = [c for c in (status - 1, status + 1) if 0 <= c < 4]
            return int(rng.choice(adjacent)), "adjacent"
        conflicting = [c for c in range(4) if abs(c - status) > 1]
        return int(rng.choice(conflicting or [3 - status])), "conflicting"

    def __getitem__(self, i: int):
        status = int(self.num["status_labels"][i])
        rng = np.random.default_rng((SEED + self.epoch * 100000 + i) % (2 ** 32))
        face_category, face_tag = self.choose_category(status, rng)
        speech_category, speech_tag = self.choose_category(status, rng)
        face_idx = int(rng.choice(self.face_by_category[face_category]))
        speech_idx = int(rng.choice(self.speech_by_category[speech_category]))
        available = torch.tensor(self.modalities, dtype=torch.bool)
        normalized_scores = self.num["regression_targets_true"][i].float() / torch.from_numpy(MAX_SCORES)
        return (
            self.face["embeddings"][face_idx].float(),
            self.speech["embeddings"][speech_idx].float(),
            self.num["embeddings"][i].float(),
            self.num["status_labels"][i].long(),
            normalized_scores,
            available,
            int(self.num["row_ids"][i]),
            face_idx, speech_idx, f"{face_tag}_{speech_tag}",
        )

