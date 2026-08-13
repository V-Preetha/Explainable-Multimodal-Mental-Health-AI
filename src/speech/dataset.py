"""Shared PyTorch Dataset/collate helpers for variable-length frame embeddings."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.speech.common import CLASS_TO_INDEX


class FrameDataset(Dataset):
    """One sample = one utterance's variable-length emotion2vec+ frame sequence."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        return row["frames"].float(), CLASS_TO_INDEX[row["emotion"]], row["path"], row["actor"]


def collate_frames(batch):
    lengths = [len(item[0]) for item in batch]
    max_len = max(lengths)
    dim = batch[0][0].shape[-1]
    frames = torch.zeros(len(batch), max_len, dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (seq, *_rest) in enumerate(batch):
        frames[i, :len(seq)] = seq
        mask[i, :len(seq)] = True
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]
    actors = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return frames, mask, labels, paths, actors


class HybridDataset(Dataset):
    """Frame sequence + matching engineered-feature vector for the same utterance."""

    def __init__(self, records: list[dict], engineered: torch.Tensor):
        assert len(records) == engineered.shape[0]
        self.records = records
        self.engineered = engineered

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        return row["frames"].float(), self.engineered[index].float(), CLASS_TO_INDEX[row["emotion"]], row["path"], row["actor"]


def collate_hybrid(batch):
    lengths = [len(item[0]) for item in batch]
    max_len = max(lengths)
    dim = batch[0][0].shape[-1]
    frames = torch.zeros(len(batch), max_len, dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (seq, *_rest) in enumerate(batch):
        frames[i, :len(seq)] = seq
        mask[i, :len(seq)] = True
    engineered = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    paths = [item[3] for item in batch]
    actors = torch.tensor([item[4] for item in batch], dtype=torch.long)
    return frames, engineered, mask, labels, paths, actors

