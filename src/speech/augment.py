"""Train-only audio augmentation for the max-accuracy random-split experiment.

Every transform is waveform-in/waveform-out at 16 kHz so it composes cleanly
before either feature extractor (emotion2vec+ or the engineered acoustic
features). SpecAugment operates on a log-mel spectrogram and is applied
separately, only to the engineered-feature branch.
"""
from __future__ import annotations

import numpy as np

from src.speech.common import SAMPLE_RATE, normalize_peak


def add_gaussian_noise(waveform: np.ndarray, rng: np.random.Generator, min_snr_db: float = 12.0, max_snr_db: float = 30.0) -> np.ndarray:
    snr_db = rng.uniform(min_snr_db, max_snr_db)
    signal_power = np.mean(waveform ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return waveform + noise


def pitch_shift(waveform: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE, semitone_range: tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    import librosa
    semitones = rng.uniform(*semitone_range)
    if abs(semitones) < 0.05:
        return waveform
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=semitones).astype(np.float32)


def time_stretch(waveform: np.ndarray, rng: np.random.Generator, rate_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    import librosa
    rate = rng.uniform(*rate_range)
    if abs(rate - 1.0) < 0.01 or waveform.size < 512:
        return waveform
    return librosa.effects.time_stretch(waveform, rate=rate).astype(np.float32)


def gain_perturbation(waveform: np.ndarray, rng: np.random.Generator, db_range: tuple[float, float] = (-6.0, 6.0)) -> np.ndarray:
    gain_db = rng.uniform(*db_range)
    return (waveform * (10 ** (gain_db / 20))).astype(np.float32)


def small_time_shift(waveform: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE, max_shift_seconds: float = 0.25) -> np.ndarray:
    max_shift = int(sr * max_shift_seconds)
    if max_shift <= 0:
        return waveform
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(waveform, shift).astype(np.float32)


def spec_augment(log_mel: np.ndarray, rng: np.random.Generator, freq_mask_width: int = 8, time_mask_width: int = 16, num_freq_masks: int = 2, num_time_masks: int = 2) -> np.ndarray:
    """log_mel shape: (n_mels, T). Masks are filled with the spectrogram mean."""
    spec = log_mel.copy()
    fill = spec.mean()
    n_mels, t = spec.shape
    for _ in range(num_freq_masks):
        width = int(rng.integers(0, freq_mask_width + 1))
        if width == 0 or width >= n_mels:
            continue
        start = int(rng.integers(0, n_mels - width))
        spec[start:start + width, :] = fill
    for _ in range(num_time_masks):
        width = int(rng.integers(0, time_mask_width + 1))
        if width == 0 or width >= t:
            continue
        start = int(rng.integers(0, t - width))
        spec[:, start:start + width] = fill
    return spec


AUGMENTATION_NAMES = ["gaussian_noise", "pitch_shift", "time_stretch", "gain", "time_shift"]


def random_waveform_augment(waveform: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE, apply_probability: float = 0.6, min_ops: int = 1, max_ops: int = 3) -> tuple[np.ndarray, list[str]]:
    """Apply a random subset (1..max_ops) of waveform augmentations independently.

    Each of the 5 transforms is independently rolled with ``apply_probability``;
    if none trigger, one is forced so augmented copies are never identical to
    the source. Output is peak-normalized to keep levels bounded.
    """
    ops = []
    chosen = [name for name in AUGMENTATION_NAMES if rng.random() < apply_probability]
    if not chosen:
        chosen = [rng.choice(AUGMENTATION_NAMES)]
    chosen = chosen[:max_ops] if len(chosen) > max_ops else chosen
    out = waveform
    for name in chosen:
        if name == "gaussian_noise":
            out = add_gaussian_noise(out, rng)
        elif name == "pitch_shift":
            out = pitch_shift(out, rng, sr)
        elif name == "time_stretch":
            out = time_stretch(out, rng)
        elif name == "gain":
            out = gain_perturbation(out, rng)
        elif name == "time_shift":
            out = small_time_shift(out, rng, sr)
        ops.append(name)
    return normalize_peak(out), ops

