"""features implementation for the curated submission repository."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.speech.common import SAMPLE_RATE, prepare_audio, root

MODEL_ID = "iic/emotion2vec_plus_base"
FRAME_DIM = 768

_ENCODER = None


def get_encoder(device: str):
    global _ENCODER
    if _ENCODER is None:
        from funasr import AutoModel
        print(f"[speech.features] loading frozen encoder {MODEL_ID} on {device} ...", flush=True)
        _ENCODER = AutoModel(model=MODEL_ID, hub="ms", device=device, disable_update=True)
    return _ENCODER


def cache_dir() -> Path:
    path = root() / "artifacts" / "cache" / "speech"
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.inference_mode()
def _embed_waveforms(model, waveforms: list[np.ndarray], batch_size: int = 12) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for start in range(0, len(waveforms), batch_size):
        batch = waveforms[start:start + batch_size]
        results = model.generate(input=batch, granularity="frame", extract_embedding=True, batch_size=len(batch))
        for result in results:
            out.append(np.asarray(result["feats"], dtype=np.float32))
    return out


def extract_real_frame_embeddings(rows: list[dict], device: str, cache_name: str = "emotion2vec_plus_frame_real.pt", batch_size: int = 12) -> dict:
    """Frame-level embeddings for real (untrimmed-content, conservatively
    trimmed) audio only -- used by both the strict benchmark and as the
    validation/test embeddings for the random-split experiment."""
    cache = cache_dir() / cache_name
    if cache.exists():
        print(f"[speech.features] reusing cached real frame embeddings: {cache}", flush=True)
        return torch.load(cache, map_location="cpu", weights_only=False)
    model = get_encoder(device)
    start_time = time.time()
    records = []
    waveforms = [prepare_audio(row["path"]) for row in rows]
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start:batch_start + batch_size]
        batch_waves = waveforms[batch_start:batch_start + batch_size]
        embeddings = _embed_waveforms(model, batch_waves, batch_size=len(batch_waves))
        for meta, frames in zip(batch_rows, embeddings):
            records.append({**meta, "frames": torch.from_numpy(frames).to(torch.float16)})
        print(f"[speech.features] emotion2vec+ frame extraction {min(batch_start + batch_size, len(rows))}/{len(rows)}", flush=True)
    payload = {"model_id": MODEL_ID, "feature_dim": FRAME_DIM, "records": records, "extraction_seconds": time.time() - start_time}
    torch.save(payload, cache)
    return payload


def extract_augmented_pair_pool(rows: list[dict], device: str, copies_per_sample: int, cache_name: str, seed: int, batch_size: int = 12) -> dict:
    """Build K augmented copies per (train-only) row and derive BOTH the
    emotion2vec+ frame embedding and the engineered feature vector from the
    exact same augmented waveform, so the two views stay paired for the
    hybrid fusion model. Returns aligned ``frame_records`` (list with a
    ``frames`` tensor each) and an ``engineered`` matrix, same length/order.
    """
    cache = cache_dir() / cache_name
    if cache.exists():
        print(f"[speech.features] reusing cached augmented pair pool: {cache}", flush=True)
        return torch.load(cache, map_location="cpu", weights_only=False)
    from src.speech.augment import random_waveform_augment

    model = get_encoder(device)
    start_time = time.time()
    meta, waveforms = [], []
    for row in rows:
        base_wave = prepare_audio(row["path"])
        for copy_index in range(copies_per_sample):
            rng = np.random.default_rng((seed + hash((row["path"], copy_index)) % (2 ** 31 - 1)) % (2 ** 32))
            augmented, ops = random_waveform_augment(base_wave, rng, SAMPLE_RATE)
            meta.append({**row, "augmentation_ops": ops, "augmentation_copy": copy_index})
            waveforms.append(augmented)

    total = len(waveforms)
    print(f"[speech.features] synthesized {total} augmented waveforms from {len(rows)} training rows x {copies_per_sample} copies", flush=True)

    frame_records = []
    for start in range(0, total, batch_size):
        batch_meta = meta[start:start + batch_size]
        batch_waves = waveforms[start:start + batch_size]
        embeddings = _embed_waveforms(model, batch_waves, batch_size=len(batch_waves))
        for m, frames in zip(batch_meta, embeddings):
            frame_records.append({**m, "frames": torch.from_numpy(frames).to(torch.float16)})
        print(f"[speech.features] augmented frame extraction {min(start + batch_size, total)}/{total}", flush=True)

    engineered_vectors = []
    for index, (m, waveform) in enumerate(zip(meta, waveforms), 1):
        spec_rng = np.random.default_rng((seed + hash((m["path"], m["augmentation_copy"], "spec")) % (2 ** 31 - 1)) % (2 ** 32))
        engineered_vectors.append(engineered_feature_vector(waveform, rng=spec_rng, apply_spec_augment=True))
        if index % 100 == 0 or index == total:
            print(f"[speech.features] augmented engineered extraction {index}/{total}", flush=True)
    engineered_matrix = np.stack(engineered_vectors).astype(np.float32)

    payload = {
        "model_id": MODEL_ID, "feature_dim": FRAME_DIM, "frame_records": frame_records,
        "engineered": torch.from_numpy(engineered_matrix), "engineered_dim": engineered_matrix.shape[1],
        "extraction_seconds": time.time() - start_time, "copies_per_sample": copies_per_sample,
    }
    torch.save(payload, cache)
    return payload


ENGINEERED_FEATURE_NAMES = [
    "mfcc", "delta_mfcc", "delta2_mfcc", "log_mel", "chroma",
    "spectral_contrast", "rms", "zcr", "pitch", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff",
]


def _stat(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)]).astype(np.float32)


def engineered_feature_vector(waveform: np.ndarray, sr: int = SAMPLE_RATE, rng: np.random.Generator | None = None, apply_spec_augment: bool = False) -> np.ndarray:
    import librosa
    from src.speech.augment import spec_augment

    mel_power = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel_power + 1e-10)
    if apply_spec_augment and rng is not None:
        log_mel = spec_augment(log_mel, rng)

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40, S=None)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
    rms = librosa.feature.rms(y=waveform)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
    f0 = librosa.yin(waveform, fmin=60, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    pitch_stats = np.array([f0.mean() if f0.size else 0.0, f0.std() if f0.size else 0.0], dtype=np.float32)

    parts = [
        _stat(mfcc), _stat(delta_mfcc), _stat(delta2_mfcc), _stat(log_mel),
        _stat(chroma), _stat(contrast), _stat(rms), _stat(zcr),
        pitch_stats, _stat(centroid), _stat(bandwidth), _stat(rolloff),
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_engineered_features(rows: list[dict], cache_name: str = "engineered_real.pt") -> dict:
    cache = cache_dir() / cache_name
    if cache.exists():
        print(f"[speech.features] reusing cached engineered features: {cache}", flush=True)
        return torch.load(cache, map_location="cpu", weights_only=False)
    vectors = []
    for index, row in enumerate(rows, 1):
        waveform = prepare_audio(row["path"])
        vectors.append(engineered_feature_vector(waveform))
        if index % 100 == 0 or index == len(rows):
            print(f"[speech.features] engineered feature extraction {index}/{len(rows)}", flush=True)
    matrix = np.stack(vectors).astype(np.float32)
    payload = {"records": rows, "features": torch.from_numpy(matrix), "feature_dim": matrix.shape[1], "feature_groups": ENGINEERED_FEATURE_NAMES}
    torch.save(payload, cache)
    return payload





