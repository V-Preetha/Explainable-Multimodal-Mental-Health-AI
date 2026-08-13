"""Shared RAVDESS loading, split, and audio-trim utilities for the speech branch.

Deliberately independent of ``src/common.py`` and ``src/models.py`` so this
branch has its own module namespace, separate from the face pipeline and the
older single-file audio scripts under ``scripts/``.
"""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

# Keep model downloads in a local gitignored cache.
_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache"
os.environ.setdefault("MODELSCOPE_CACHE", str(_CACHE_ROOT / "modelscope"))
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))

import numpy as np
import soundfile as sf
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SAMPLE_RATE = 16000

# RAVDESS filename field 3 -> emotion label (speech-only set, 8 classes).
EMOTION_CODE = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
CLASSES = sorted(EMOTION_CODE.values())
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Actor-independent split preserved from the existing audio pipeline
# (see scripts/train_audio_emotion2vec.py and README integrity rules):
# 16 train / 4 validation / 4 test actors, zero overlap.
TRAIN_ACTORS = [1, 4, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24]
VAL_ACTORS = [3, 5, 15, 23]
TEST_ACTORS = [2, 9, 14, 22]


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    value = os.environ.get("AEGIS_DATA_ROOT")
    if not value:
        raise RuntimeError("Set AEGIS_DATA_ROOT to the directory containing the local datasets.")
    return Path(value).expanduser().resolve()


def audio_dir() -> Path:
    return data_root() / "Audios"


def assert_zero_actor_overlap() -> None:
    train, val, test = set(TRAIN_ACTORS), set(VAL_ACTORS), set(TEST_ACTORS)
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise RuntimeError(f"Actor overlap detected between splits: {sorted(overlap)}")
    if len(train) != 16 or len(val) != 4 or len(test) != 4:
        raise RuntimeError(f"Unexpected split sizes: train={len(train)} val={len(val)} test={len(test)}")


def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ravdess_rows(verbose: bool = True) -> list[dict]:
    """Walk the RAVDESS Audios tree, parse metadata, and deduplicate mirrored copies.

    Two layers of dedup: (1) identical filename already seen, (2) identical
    file content hash under a different name/location (a true mirrored copy).
    Only well-formed 7-field RAVDESS filenames are kept; only speech-modality
    (field 1 == "03") vocal-channel files are considered.
    """
    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    rows: list[dict] = []
    duplicate_names, duplicate_hashes, malformed = 0, 0, 0
    for path in sorted(audio_dir().rglob("*.wav")):
        fields = path.stem.split("-")
        if len(fields) != 7 or fields[2] not in EMOTION_CODE:
            malformed += 1
            continue
        if path.name in seen_names:
            duplicate_names += 1
            continue
        seen_names.add(path.name)
        digest = _file_hash(path)
        if digest in seen_hashes:
            duplicate_hashes += 1
            continue
        seen_hashes.add(digest)
        rows.append({
            "path": str(path),
            "modality": fields[0],
            "vocal_channel": fields[1],
            "emotion": EMOTION_CODE[fields[2]],
            "intensity": fields[3],
            "statement": fields[4],
            "repetition": fields[5],
            "actor": int(fields[6]),
            "sha256": digest,
        })
    if verbose:
        print(
            f"[speech.common] loaded {len(rows)} unique real audio files "
            f"(dropped {duplicate_names} duplicate filenames, "
            f"{duplicate_hashes} duplicate-content mirrors, {malformed} malformed names)",
            flush=True,
        )
    return rows


def split_indices(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert_zero_actor_overlap()
    actors = np.asarray([row["actor"] for row in rows])
    train_idx = np.flatnonzero(np.isin(actors, TRAIN_ACTORS))
    val_idx = np.flatnonzero(np.isin(actors, VAL_ACTORS))
    test_idx = np.flatnonzero(np.isin(actors, TEST_ACTORS))
    overlap = (set(train_idx) & set(val_idx)) | (set(train_idx) & set(test_idx)) | (set(val_idx) & set(test_idx))
    if overlap:
        raise RuntimeError("Sample-index overlap between splits despite disjoint actor lists.")
    return train_idx, val_idx, test_idx


def load_waveform(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    waveform, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if file_sr != sr:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=sr)
    return waveform.astype(np.float32)


def conservative_trim(waveform: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 22.0) -> np.ndarray:
    """Trim only leading/trailing near-silence; preserve internal pauses.

    ``librosa.effects.trim`` finds the first and last frame above the energy
    threshold and cuts once at each end -- it never removes audio between
    those two points, so pauses/silences inside an utterance are untouched.
    A conservative (fairly low) top_db keeps quiet speech onsets/offsets intact.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db, frame_length=1024, hop_length=256)
    if trimmed.size < sr * 0.2:
        return waveform
    return trimmed


def normalize_peak(waveform: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(waveform)) if waveform.size else 0.0
    if peak > 1e-8:
        return waveform / peak
    return waveform


def prepare_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    waveform = load_waveform(path, sr)
    waveform = conservative_trim(waveform, sr)
    return normalize_peak(waveform)

