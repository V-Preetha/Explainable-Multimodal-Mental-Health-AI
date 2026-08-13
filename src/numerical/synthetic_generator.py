"""Synthetic-enhanced numerical dataset generator.

Causal structure (deliberately layered so no single feature is a near-
perfect classifier and Mental_Health_Status is never a direct threshold of
Depression+Anxiety+Stress):

    class label (balanced, assigned first)
        -> latent severity in [0,1]   (overlapping Beta ranges per class)
        -> shared distress axis G     (correlated with severity, own noise)
        -> 7 domain latent factors    (weighted combo of severity + G + own noise)
            sleep_disruption, social_withdrawal, digital_dysregulation,
            negative_affect, speech_instability, autonomic_arousal,
            behavioral_irregularity
        -> depression_factor / anxiety_factor / stress_factor
           (differentiated weighted combos of the 7 latents + G + own noise --
            NOT derived from each other, NOT derived from status directly)
        -> Depression_Score / Anxiety_Score / Stress_Score
           (factor * max_score + noise, clipped to the real dataset's range)
        -> 18 original + 38 new raw features
           (each a domain-appropriate weighted combo of the *7 latents only*
            -- never of the D/A/S factors or of status directly -- plus
            substantial independent per-row noise)
        -> 15 derived features (arithmetic combinations of the raw features,
           documented per-formula below, epsilon-guarded against divide-by-zero)

Every raw/derived feature blends its latent-driven component with
substantial independent noise (and, for several, an explicit random-blend
weight < 1 against the latent) specifically so that no single column
approaches deterministic separability -- see ``numerical_synthetic/audit.py``
for the checks that verify this held.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.numerical.synthetic_common import (
    ALL_FEATURES, AUDIT_LATENT_PREFIX, BEHAVIORAL_FEATURES, DERIVED_FEATURES,
    DIGITAL_FEATURES, FACIAL_FEATURES, NEW_RAW_FEATURES, ORIGINAL_FEATURES,
    PHYSIO_FEATURES, PROVENANCE_COLUMN, REGRESSION_TARGETS, SEED,
    SEED_COLUMN, SLEEP_FEATURES, SOCIAL_FEATURES, SPEECH_FEATURES,
    STATUS_CLASSES, data_dir, data_root, out_root,
)

EPS = 1e-6
MAX_SCORES = {"Depression_Score": 34, "Anxiety_Score": 24, "Stress_Score": 39}

# Overlapping per-class severity ranges (substantial overlap at boundaries).
SEVERITY_RANGES = {
    "Healthy": (0.00, 0.30),
    "Mild_Stress": (0.20, 0.50),
    "Moderate_Stress": (0.40, 0.75),
    "Severe_Stress": (0.65, 1.00),
}

# Weighted contribution of each of the 7 domain latents into the shared axis
# G is NOT a linear combination of the 7 latents (that would be circular --
# G is generated first, the 7 latents are generated FROM severity+G). Instead
# G is its own severity-correlated axis, see generate_latents().


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def sample_severity(status_labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    severity = np.empty(len(status_labels), dtype=np.float64)
    for status_name, (lo, hi) in SEVERITY_RANGES.items():
        mask = status_labels == status_name
        n = int(mask.sum())
        if n == 0:
            continue
        # Beta(10, 10) concentrates most mass mid-range within each class's
        # own severity band while still reaching both edges -- calibrated
        # (see numerical_synthetic/audit.py sweep in development) so the
        # boundary overlap with neighboring classes stays substantial without
        # making the overall multi-feature classification task trivial.
        beta = rng.beta(10.0, 10.0, size=n)
        severity[mask] = lo + (hi - lo) * beta
    return severity


def generate_latents(severity: np.ndarray, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n = len(severity)
    # Shared distress axis: correlated with severity but with its own
    # substantial noise, so it is not a deterministic function of severity.
    shared_axis = clip01(0.6 * severity + 0.4 * rng.normal(0.5, 0.22, n))

    def factor(severity_weight: float, shared_weight: float, noise_std: float) -> np.ndarray:
        return clip01(severity_weight * severity + shared_weight * shared_axis + rng.normal(0.0, noise_std, n))

    # noise_std values calibrated (see development sweep) to keep the
    # multi-feature classification ceiling in the ~85-95% band -- the raw
    # feature formulas below add substantial additional independent noise on
    # top of this, so this only controls how precisely the latent
    # *constructs* themselves reflect class-conditional severity.
    latents = {
        "sleep_disruption": factor(0.45, 0.35, 0.027),
        "social_withdrawal": factor(0.40, 0.30, 0.029),
        "digital_dysregulation": factor(0.35, 0.35, 0.032),
        "negative_affect": factor(0.50, 0.30, 0.027),
        "speech_instability": factor(0.35, 0.25, 0.036),
        "autonomic_arousal": factor(0.45, 0.35, 0.027),
        "behavioral_irregularity": factor(0.40, 0.30, 0.032),
    }
    return latents, shared_axis


DEPRESSION_WEIGHTS = {"sleep_disruption": 0.28, "social_withdrawal": 0.26, "digital_dysregulation": 0.14, "negative_affect": 0.22, "speech_instability": 0.04, "autonomic_arousal": 0.03, "behavioral_irregularity": 0.08}
ANXIETY_WEIGHTS = {"sleep_disruption": 0.08, "social_withdrawal": 0.06, "digital_dysregulation": 0.06, "negative_affect": 0.14, "speech_instability": 0.22, "autonomic_arousal": 0.34, "behavioral_irregularity": 0.10}
STRESS_WEIGHTS = {"sleep_disruption": 0.20, "social_withdrawal": 0.06, "digital_dysregulation": 0.15, "negative_affect": 0.09, "speech_instability": 0.12, "autonomic_arousal": 0.18, "behavioral_irregularity": 0.20}


def _weighted(latents: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total_weight = sum(weights.values())
    return sum(latents[name] * (weight / total_weight) for name, weight in weights.items())


def generate_dsa_factors(latents: dict[str, np.ndarray], shared_axis: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(shared_axis)
    # Each factor = 0.85 * (its own weighted combination of the 7 latents)
    #             + 0.15 * shared_axis   (shared variance across all 3, per spec item 9)
    #             + independent noise
    depression_factor = clip01(0.85 * _weighted(latents, DEPRESSION_WEIGHTS) + 0.15 * shared_axis + rng.normal(0.0, 0.07, n))
    anxiety_factor = clip01(0.85 * _weighted(latents, ANXIETY_WEIGHTS) + 0.15 * shared_axis + rng.normal(0.0, 0.07, n))
    stress_factor = clip01(0.85 * _weighted(latents, STRESS_WEIGHTS) + 0.15 * shared_axis + rng.normal(0.0, 0.07, n))
    return {"depression_factor": depression_factor, "anxiety_factor": anxiety_factor, "stress_factor": stress_factor}


def scores_from_factors(factors: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(factors["depression_factor"])
    depression = np.clip(np.round(factors["depression_factor"] * MAX_SCORES["Depression_Score"] + rng.normal(0, 2.0, n)), 0, MAX_SCORES["Depression_Score"])
    anxiety = np.clip(np.round(factors["anxiety_factor"] * MAX_SCORES["Anxiety_Score"] + rng.normal(0, 1.5, n)), 0, MAX_SCORES["Anxiety_Score"])
    stress = np.clip(np.round(factors["stress_factor"] * MAX_SCORES["Stress_Score"] + rng.normal(0, 2.2, n)), 0, MAX_SCORES["Stress_Score"])
    return {"Depression_Score": depression.astype(int), "Anxiety_Score": anxiety.astype(int), "Stress_Score": stress.astype(int)}


def generate_original_style_features(latents: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(next(iter(latents.values())))
    sd, sw, dd, na, si, aa, bi = (latents[k] for k in ("sleep_disruption", "social_withdrawal", "digital_dysregulation", "negative_affect", "speech_instability", "autonomic_arousal", "behavioral_irregularity"))
    out = {}
    out["Sleep_Quality"] = np.clip(np.round(5 - sd * 3.5 + rng.normal(0, 0.8, n)), 1, 5).astype(int)
    out["Social_Engagement"] = np.clip(np.round(5 - sw * 3.5 + rng.normal(0, 0.8, n)), 1, 5).astype(int)
    out["Daily_App_Usage_Min"] = np.clip(np.round(150 + dd * 250 + rng.normal(0, 60, n)), 30, 479).astype(int)
    out["Typing_Speed_WPM"] = np.clip(np.round(75 - (0.5 * na + 0.5 * sd) * 40 + rng.normal(0, 8, n)), 20, 89).astype(int)
    out["Session_Frequency"] = np.clip(np.round(5 + dd * 12 + rng.normal(0, 2.5, n)), 1, 19).astype(int)
    out["Idle_Time_Min"] = np.clip(np.round(40 + (0.5 * bi + 0.5 * sw) * 120 + rng.normal(0, 20, n)), 5, 179).astype(int)
    out["Facial_Emotion_Variance"] = np.clip(0.3 + na * 0.5 + rng.normal(0, 0.12, n), 0.1, 1.0)
    out["Eye_Blink_Rate"] = np.clip(np.round(15 + aa * 15 + rng.normal(0, 3, n)), 10, 34).astype(int)
    out["Smile_Intensity"] = np.clip(0.85 - na * 0.7 + rng.normal(0, 0.12, n), 0.0, 1.0)
    out["Head_Motion_Index"] = np.clip(0.25 + (0.5 * aa + 0.5 * bi) * 0.6 + rng.normal(0, 0.12, n), 0.0, 1.0)
    out["MFCC_Mean"] = np.clip(-5 * na + rng.normal(0, 20, n), -50, 50)
    out["MFCC_Variance"] = np.clip(8 + si * 15 + rng.normal(0, 4, n), 1, 30)
    out["Pitch_Mean"] = np.clip(150 + aa * 80 + rng.normal(0, 30, n), 80, 300)
    out["Speech_Rate"] = np.clip(4 + aa * 1.5 - sd * 0.3 + rng.normal(0, 0.6, n), 2, 6)
    out["Heart_Rate_BPM"] = np.clip(np.round(68 + aa * 40 + rng.normal(0, 7, n)), 55, 119).astype(int)
    out["HRV_Index"] = np.clip(85 - aa * 65 + rng.normal(0, 10, n), 10, 100)
    out["Skin_Temperature"] = np.clip(35.2 - aa * 1.5 + rng.normal(0, 0.5, n), 32, 37)
    out["GSR_Level"] = np.clip(1.0 + aa * 3.2 + rng.normal(0, 0.6, n), 0.1, 5)
    return out


def generate_new_raw_features(latents: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(next(iter(latents.values())))
    sd, sw, dd, na, si, aa, bi = (latents[k] for k in ("sleep_disruption", "social_withdrawal", "digital_dysregulation", "negative_affect", "speech_instability", "autonomic_arousal", "behavioral_irregularity"))
    out = {}
    # Sleep
    out["Sleep_Duration_Hours"] = np.clip(7.5 - sd * 2.5 + rng.normal(0, 0.6, n), 3, 10)
    out["Sleep_Onset_Latency_Min"] = np.clip(10 + sd * 40 + rng.normal(0, 8, n), 2, 120)
    out["Night_Awakenings"] = np.clip(np.round(sd * 4 + rng.normal(0, 1, n)), 0, 8).astype(int)
    out["Sleep_Regularity_Index"] = np.clip(0.9 - sd * 0.6 + rng.normal(0, 0.1, n), 0, 1)
    out["Wake_Time_Variability_Min"] = np.clip(15 + sd * 90 + rng.normal(0, 15, n), 0, 240)
    # Digital behavior
    out["App_Usage_Variability"] = np.clip(0.2 + dd * 0.6 + rng.normal(0, 0.1, n), 0, 1)
    out["Night_App_Usage_Min"] = np.clip(5 + dd * 80 + sd * 20 + rng.normal(0, 15, n), 0, 240)
    out["Average_Session_Duration_Min"] = np.clip(3 + dd * 12 + rng.normal(0, 3, n), 0.5, 40)
    out["Typing_Error_Rate"] = np.clip(0.03 + na * 0.15 + sd * 0.05 + rng.normal(0, 0.03, n), 0, 0.6)
    out["Typing_Latency_ms"] = np.clip(150 + na * 250 + rng.normal(0, 50, n), 80, 900)
    out["Screen_Unlock_Count"] = np.clip(np.round(20 + dd * 100 + rng.normal(0, 15, n)), 5, 300).astype(int)
    # Social behavior
    out["Social_Interaction_Frequency"] = np.clip(8 - sw * 6.5 + rng.normal(0, 1.5, n), 0, 15)
    out["Social_Withdrawal_Index"] = np.clip(0.7 * sw + 0.3 * rng.random(n) + rng.normal(0, 0.1, n), 0, 1)
    out["Communication_Response_Delay_Min"] = np.clip(5 + sw * 45 + rng.normal(0, 10, n), 1, 180)
    # Facial behavior
    out["Negative_Emotion_Ratio"] = np.clip(0.15 + na * 0.6 + rng.normal(0, 0.1, n), 0, 1)
    out["Positive_Emotion_Ratio"] = np.clip(0.75 - na * 0.55 + rng.normal(0, 0.1, n), 0, 1)
    out["Emotion_Transition_Rate"] = np.clip(2 + na * 3 + si * 1 + rng.normal(0, 0.8, n), 0, 10)
    out["Gaze_Avoidance_Index"] = np.clip(0.15 + aa * 0.5 + sw * 0.2 + rng.normal(0, 0.1, n), 0, 1)
    out["Facial_Tension_Index"] = np.clip(0.15 + aa * 0.4 + bi * 0.2 + rng.normal(0, 0.1, n), 0, 1)
    # Speech/acoustic
    out["Pitch_Variance"] = np.clip(10 + si * 40 + rng.normal(0, 8, n), 1, 100)
    out["Energy_Mean"] = np.clip(0.5 + aa * 0.2 - sd * 0.1 + rng.normal(0, 0.1, n), 0, 1)
    out["Energy_Variance"] = np.clip(0.05 + si * 0.25 + rng.normal(0, 0.05, n), 0, 1)
    out["Pause_Rate"] = np.clip(5 + si * 8 + sd * 3 + rng.normal(0, 2, n), 0, 30)
    out["Average_Pause_Duration"] = np.clip(0.4 + si * 0.8 + rng.normal(0, 0.2, n), 0.1, 4)
    out["Jitter"] = np.clip(0.3 + si * 1.2 + rng.normal(0, 0.2, n), 0.05, 3)
    out["Shimmer"] = np.clip(2 + si * 6 + rng.normal(0, 1, n), 0.5, 15)
    out["Voice_Stability_Index"] = np.clip(0.9 - si * 0.6 + rng.normal(0, 0.08, n), 0, 1)
    # Physiological
    out["Resting_Heart_Rate"] = np.clip(65 + aa * 25 + rng.normal(0, 6, n), 50, 110)
    out["RMSSD"] = np.clip(45 - aa * 30 + rng.normal(0, 7, n), 5, 90)
    out["SDNN"] = np.clip(55 - aa * 35 + rng.normal(0, 8, n), 5, 110)
    out["LF_HF_Ratio"] = np.clip(1.2 + aa * 2.5 + rng.normal(0, 0.5, n), 0.2, 6)
    out["GSR_Mean"] = np.clip(1.0 + aa * 3 + rng.normal(0, 0.5, n), 0.1, 5)
    out["GSR_Variance"] = np.clip(0.1 + aa * 0.8 + rng.normal(0, 0.15, n), 0.01, 2)
    out["GSR_Peaks_Per_Min"] = np.clip(2 + aa * 10 + rng.normal(0, 2, n), 0, 25)
    out["Skin_Temperature_Variability"] = np.clip(0.1 + aa * 0.5 + rng.normal(0, 0.1, n), 0.02, 1.5)
    # Behavioral/circadian
    out["Activity_Index"] = np.clip(0.75 - bi * 0.45 - sd * 0.15 + rng.normal(0, 0.1, n), 0, 1)
    out["Circadian_Disruption_Index"] = np.clip(0.15 + sd * 0.5 + bi * 0.25 + rng.normal(0, 0.1, n), 0, 1)
    out["Daily_Routine_Consistency"] = np.clip(0.85 - bi * 0.6 + rng.normal(0, 0.1, n), 0, 1)
    return out


def generate_derived_features(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every formula documented inline; all divisions are epsilon-guarded."""
    out = {}
    # Autonomic ratio: resting/arousal heart rate relative to vagal tone (HRV_Index).
    out["HR_HRV_Ratio"] = raw["Heart_Rate_BPM"] / (raw["HRV_Index"] + EPS)
    # Simple interaction of cardiac load and electrodermal arousal.
    out["HR_GSR_Interaction"] = (raw["Heart_Rate_BPM"] / 100.0) * raw["GSR_Level"]
    # Electrodermal activity relative to short-term HRV (SDNN): high GSR + low SDNN = high index.
    out["GSR_HRV_Stress_Index"] = raw["GSR_Mean"] / (raw["SDNN"] + EPS)
    # Hours short of the commonly-cited 8h sleep recommendation (floored at 0).
    out["Sleep_Deficit"] = np.clip(8.0 - raw["Sleep_Duration_Hours"], 0, None)
    # Awakenings per hour slept.
    out["Sleep_Fragmentation_Index"] = raw["Night_Awakenings"] / (raw["Sleep_Duration_Hours"] + EPS)
    # Fraction of total app usage that happens at night.
    out["Night_Usage_Ratio"] = raw["Night_App_Usage_Min"] / (raw["Daily_App_Usage_Min"] + EPS)
    # Average minutes spent per app session.
    out["App_Usage_Per_Session"] = raw["Daily_App_Usage_Min"] / (raw["Session_Frequency"] + EPS)
    # Idle time as a fraction of total (idle + active-usage) time.
    out["Idle_Usage_Ratio"] = raw["Idle_Time_Min"] / (raw["Daily_App_Usage_Min"] + raw["Idle_Time_Min"] + EPS)
    # Blend of self-report-style withdrawal index and normalized (inverse) interaction frequency.
    out["Social_Withdrawal_Score"] = 0.5 * raw["Social_Withdrawal_Index"] + 0.5 * (1 - np.clip(raw["Social_Interaction_Frequency"] / 15.0, 0, 1))
    # Composite of jitter/shimmer (normalized to their plausible max) and inverse voice stability.
    out["Speech_Instability_Index"] = 0.4 * np.clip(raw["Jitter"] / 3.0, 0, 1) + 0.3 * np.clip(raw["Shimmer"] / 15.0, 0, 1) + 0.3 * (1 - raw["Voice_Stability_Index"])
    # Normalized pitch x speech-rate interaction.
    out["Pitch_SpeechRate_Interaction"] = (raw["Pitch_Mean"] / 200.0) * (raw["Speech_Rate"] / 4.0)
    # Blend of negative-emotion ratio and inverse smile intensity.
    out["Negative_Expression_Index"] = 0.6 * raw["Negative_Emotion_Ratio"] + 0.4 * (1 - raw["Smile_Intensity"])
    # Blend of gaze avoidance and facial tension.
    out["Facial_Arousal_Index"] = 0.5 * raw["Gaze_Avoidance_Index"] + 0.5 * raw["Facial_Tension_Index"]
    # Composite of four normalized autonomic-nervous-system raw signals.
    out["Autonomic_Arousal_Index"] = 0.25 * np.clip(raw["Heart_Rate_BPM"] / 100.0, 0, None) + 0.25 * np.clip(1 - raw["HRV_Index"] / 100.0, 0, None) + 0.25 * np.clip(raw["GSR_Level"] / 5.0, 0, None) + 0.25 * np.clip(raw["LF_HF_Ratio"] / 6.0, 0, None)
    # Composite of circadian, routine, digital, and sleep-fragmentation disruption signals.
    out["Overall_Behavioral_Disruption_Index"] = 0.25 * raw["Circadian_Disruption_Index"] + 0.25 * (1 - raw["Daily_Routine_Consistency"]) + 0.25 * raw["App_Usage_Variability"] + 0.25 * np.clip(out["Sleep_Fragmentation_Index"] / 2.0, 0, 1)
    return out


def generate_synthetic_dataset(n_rows: int, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    per_class = n_rows // len(STATUS_CLASSES)
    status_labels = np.repeat(STATUS_CLASSES, per_class)
    if len(status_labels) < n_rows:
        # top up with extra Healthy rows if n_rows isn't divisible by 4
        status_labels = np.concatenate([status_labels, np.repeat(STATUS_CLASSES[0], n_rows - len(status_labels))])
    rng.shuffle(status_labels)

    severity = sample_severity(status_labels, rng)
    latents, shared_axis = generate_latents(severity, rng)
    dsa_factors = generate_dsa_factors(latents, shared_axis, rng)
    scores = scores_from_factors(dsa_factors, rng)
    original_style = generate_original_style_features(latents, rng)
    new_raw = generate_new_raw_features(latents, rng)
    all_raw = {**original_style, **new_raw}
    derived = generate_derived_features(all_raw)

    columns = {
        PROVENANCE_COLUMN: np.repeat("synthetic", n_rows),
        SEED_COLUMN: np.repeat(seed, n_rows),
        f"{AUDIT_LATENT_PREFIX}severity": severity,
        **{f"{AUDIT_LATENT_PREFIX}{name}": values for name, values in latents.items()},
        **{f"{AUDIT_LATENT_PREFIX}{name}": values for name, values in dsa_factors.items()},
        **all_raw,
        **derived,
        **scores,
        "Mental_Health_Status": status_labels,
    }
    df = pd.DataFrame(columns)
    df = df[[PROVENANCE_COLUMN, SEED_COLUMN] + [c for c in df.columns if c.startswith(AUDIT_LATENT_PREFIX)] + ALL_FEATURES + REGRESSION_TARGETS + ["Mental_Health_Status"]]

    config = {
        "seed": seed, "n_rows": int(n_rows),
        "class_counts": {name: int((status_labels == name).sum()) for name in STATUS_CLASSES},
        "severity_ranges": SEVERITY_RANGES,
        "severity_distribution": "per-class Beta(10,10) rescaled into that class's [lo,hi] range",
        "latent_factor_generation": "shared_axis = clip01(0.6*severity + 0.4*N(0.5,0.22)); each of the 7 domain latents = clip01(severity_weight*severity + shared_weight*shared_axis + N(0,noise_std)); see generate_latents() for exact per-factor weights/noise_std (noise_std ~0.027-0.036, calibrated via an offline accuracy/leakage sweep so the resulting multi-feature classification ceiling lands near 85-95% while no single feature exceeds ~0.65 accuracy)",
        "dsa_factor_weights": {"depression": DEPRESSION_WEIGHTS, "anxiety": ANXIETY_WEIGHTS, "stress": STRESS_WEIGHTS},
        "dsa_factor_generation": "factor = clip01(0.85 * normalized_weighted_sum(7 latents) + 0.15 * shared_axis + N(0, 0.07)); Depression/Anxiety/Stress scores = clip(round(factor * max_score + N(0, score_noise_std)), 0, max_score); score_noise_std = {Depression: 2.0, Anxiety: 1.5, Stress: 2.2}",
        "max_scores": MAX_SCORES,
        "feature_ranges": {col: {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean()), "std": float(df[col].std())} for col in ALL_FEATURES},
        "derived_feature_formulas": {
            "HR_HRV_Ratio": "Heart_Rate_BPM / (HRV_Index + eps)",
            "HR_GSR_Interaction": "(Heart_Rate_BPM / 100) * GSR_Level",
            "GSR_HRV_Stress_Index": "GSR_Mean / (SDNN + eps)",
            "Sleep_Deficit": "max(0, 8 - Sleep_Duration_Hours)",
            "Sleep_Fragmentation_Index": "Night_Awakenings / (Sleep_Duration_Hours + eps)",
            "Night_Usage_Ratio": "Night_App_Usage_Min / (Daily_App_Usage_Min + eps)",
            "App_Usage_Per_Session": "Daily_App_Usage_Min / (Session_Frequency + eps)",
            "Idle_Usage_Ratio": "Idle_Time_Min / (Daily_App_Usage_Min + Idle_Time_Min + eps)",
            "Social_Withdrawal_Score": "0.5*Social_Withdrawal_Index + 0.5*(1 - clip(Social_Interaction_Frequency/15, 0, 1))",
            "Speech_Instability_Index": "0.4*clip(Jitter/3,0,1) + 0.3*clip(Shimmer/15,0,1) + 0.3*(1 - Voice_Stability_Index)",
            "Pitch_SpeechRate_Interaction": "(Pitch_Mean/200) * (Speech_Rate/4)",
            "Negative_Expression_Index": "0.6*Negative_Emotion_Ratio + 0.4*(1 - Smile_Intensity)",
            "Facial_Arousal_Index": "0.5*Gaze_Avoidance_Index + 0.5*Facial_Tension_Index",
            "Autonomic_Arousal_Index": "0.25*clip(HR/100,0,None) + 0.25*clip(1-HRV_Index/100,0,None) + 0.25*clip(GSR_Level/5,0,None) + 0.25*clip(LF_HF_Ratio/6,0,None)",
            "Overall_Behavioral_Disruption_Index": "0.25*Circadian_Disruption_Index + 0.25*(1-Daily_Routine_Consistency) + 0.25*App_Usage_Variability + 0.25*clip(Sleep_Fragmentation_Index/2,0,1)",
        },
        "epsilon": EPS,
        "leakage_avoidance": [
            "Mental_Health_Status is assigned first (balanced) and drives severity, not the reverse.",
            "Depression/Anxiety/Stress factors are computed from the 7 shared latents independently of each other and of status -- never as a threshold/sum of one another.",
            "All 18 original + 38 new raw features are computed only from the 7 domain latents (never from the D/A/S factors, never from status, never from each other's derived values).",
            "audit_latent_* columns are never included in ALL_FEATURES / model inputs.",
        ],
    }
    return df, config


def main(n_rows: int = 20000, seed: int = SEED) -> None:
    started = time.time()
    print(f"[GEN] generating {n_rows} synthetic rows (seed={seed}) ...", flush=True)
    synthetic_df, config = generate_synthetic_dataset(n_rows, seed)
    data_dir().mkdir(parents=True, exist_ok=True)
    synthetic_path = data_dir() / "synthetic_numerical_20000.csv"
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"[GEN] saved synthetic dataset -> {synthetic_path} ({len(synthetic_df)} rows, {synthetic_df.shape[1]} columns)", flush=True)

    original_df = pd.read_csv(data_root() / "mental_health_multimodal.csv")
    original_df[PROVENANCE_COLUMN] = "original"
    original_df[SEED_COLUMN] = -1
    # Real participants were only ever measured on the 18 original features.
    # The 38 new + 15 derived features (and the audit_latent_* columns) do
    # not exist for them -- left as NaN here (never silently imputed inside
    # the raw CSV); numerical_synthetic/dataset_regimes.py imputes them from
    # the synthetic-training median only at model-input construction time,
    # and documents that this makes those columns uninformative (constant)
    # for original-provenance rows, by design.
    missing_columns = [c for c in synthetic_df.columns if c not in original_df.columns]
    for col in missing_columns:
        original_df[col] = np.nan
    original_df = original_df[synthetic_df.columns]
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    combined_path = data_dir() / "combined_original_synthetic.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"[GEN] saved combined dataset -> {combined_path} ({len(combined_df)} rows: {len(original_df)} original + {len(synthetic_df)} synthetic)", flush=True)

    config["generation_seconds"] = time.time() - started
    config["combined_row_count"] = int(len(combined_df))
    config["original_row_count"] = int(len(original_df))
    config_path = data_dir() / "generator_config.json"
    config_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"[GEN] saved generator config -> {config_path}", flush=True)
    print(f"[GEN] class counts: {config['class_counts']}", flush=True)


if __name__ == "__main__":
    main()


