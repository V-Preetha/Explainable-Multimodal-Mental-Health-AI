"""synthetic common implementation for the curated submission repository."""
from __future__ import annotations

from pathlib import Path

from src.numerical.common import (  # noqa: F401  (re-exported for this branch)
    FEATURES as ORIGINAL_FEATURES,
    NUM_CLASSES,
    REGRESSION_TARGETS,
    SEED,
    STATUS_CLASSES,
    STATUS_TO_INDEX,
    data_root,
    root,
)

AUDIT_LATENT_PREFIX = "audit_latent_"
PROVENANCE_COLUMN = "source"
SEED_COLUMN = "generation_seed"

# --- newly generated raw synthetic features, grouped by domain (38 total) ---
SLEEP_FEATURES = [
    "Sleep_Duration_Hours", "Sleep_Onset_Latency_Min", "Night_Awakenings",
    "Sleep_Regularity_Index", "Wake_Time_Variability_Min",
]
DIGITAL_FEATURES = [
    "App_Usage_Variability", "Night_App_Usage_Min", "Average_Session_Duration_Min",
    "Typing_Error_Rate", "Typing_Latency_ms", "Screen_Unlock_Count",
]
SOCIAL_FEATURES = [
    "Social_Interaction_Frequency", "Social_Withdrawal_Index", "Communication_Response_Delay_Min",
]
FACIAL_FEATURES = [
    "Negative_Emotion_Ratio", "Positive_Emotion_Ratio", "Emotion_Transition_Rate",
    "Gaze_Avoidance_Index", "Facial_Tension_Index",
]
SPEECH_FEATURES = [
    "Pitch_Variance", "Energy_Mean", "Energy_Variance", "Pause_Rate",
    "Average_Pause_Duration", "Jitter", "Shimmer", "Voice_Stability_Index",
]
PHYSIO_FEATURES = [
    "Resting_Heart_Rate", "RMSSD", "SDNN", "LF_HF_Ratio", "GSR_Mean",
    "GSR_Variance", "GSR_Peaks_Per_Min", "Skin_Temperature_Variability",
]
BEHAVIORAL_FEATURES = [
    "Activity_Index", "Circadian_Disruption_Index", "Daily_Routine_Consistency",
]
NEW_RAW_FEATURES = (
    SLEEP_FEATURES + DIGITAL_FEATURES + SOCIAL_FEATURES + FACIAL_FEATURES
    + SPEECH_FEATURES + PHYSIO_FEATURES + BEHAVIORAL_FEATURES
)

# --- derived features computed from raw features (15 total); formulas are
# documented at their point of computation in generator.py ---
DERIVED_FEATURES = [
    "HR_HRV_Ratio", "HR_GSR_Interaction", "GSR_HRV_Stress_Index",
    "Sleep_Deficit", "Sleep_Fragmentation_Index",
    "Night_Usage_Ratio", "App_Usage_Per_Session", "Idle_Usage_Ratio",
    "Social_Withdrawal_Score",
    "Speech_Instability_Index", "Pitch_SpeechRate_Interaction",
    "Negative_Expression_Index", "Facial_Arousal_Index",
    "Autonomic_Arousal_Index",
    "Overall_Behavioral_Disruption_Index",
]

# The full input feature list every model in this branch trains on.
ALL_FEATURES = list(ORIGINAL_FEATURES) + list(NEW_RAW_FEATURES) + list(DERIVED_FEATURES)

# Latent/audit columns: retained in the saved CSVs for auditing, prefixed so
# they can never be silently included as model inputs by a glob/prefix match.
LATENT_FACTOR_NAMES = [
    "severity", "sleep_disruption", "social_withdrawal", "digital_dysregulation",
    "negative_affect", "speech_instability", "autonomic_arousal", "behavioral_irregularity",
    "depression_factor", "anxiety_factor", "stress_factor",
]
AUDIT_COLUMNS = [f"{AUDIT_LATENT_PREFIX}{name}" for name in LATENT_FACTOR_NAMES]

NON_FEATURE_COLUMNS = [PROVENANCE_COLUMN, SEED_COLUMN, "Mental_Health_Status"] + REGRESSION_TARGETS + AUDIT_COLUMNS


def out_root() -> Path:
    path = root() / "artifacts" / "numerical_synthetic"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = out_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plots_dir() -> Path:
    path = out_root() / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = out_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_dir() -> Path:
    path = out_root() / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_no_leakage(feature_list: list[str]) -> None:
    banned = set(NON_FEATURE_COLUMNS)
    leaked = [f for f in feature_list if f in banned or f.startswith(AUDIT_LATENT_PREFIX)]
    if leaked:
        raise RuntimeError(f"Refusing to train: leaked non-feature columns in feature list: {leaked}")




