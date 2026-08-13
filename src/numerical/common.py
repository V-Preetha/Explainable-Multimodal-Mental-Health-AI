from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import SEED, data_root, dump_json, root

FEATURES = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", "Typing_Speed_WPM",
    "Session_Frequency", "Idle_Time_Min", "Facial_Emotion_Variance", "Eye_Blink_Rate",
    "Smile_Intensity", "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", "Pitch_Mean",
    "Speech_Rate", "Heart_Rate_BPM", "HRV_Index", "Skin_Temperature", "GSR_Level",
]
REGRESSION_TARGETS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
STATUS_CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
STATUS_TO_INDEX = {name: index for index, name in enumerate(STATUS_CLASSES)}
NUM_CLASSES = len(STATUS_CLASSES)


def load_dataframe() -> pd.DataFrame:
    frame = pd.read_csv(data_root() / "mental_health_multimodal.csv")
    required = FEATURES + REGRESSION_TARGETS + ["Mental_Health_Status"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"Missing required numerical columns: {missing}")
    return frame


def load_fixed_split():
    path = root() / "configs" / "numerical_split.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    arrays = tuple(np.asarray(payload[name], dtype=np.int64) for name in ("train", "validation", "test"))
    if (set(arrays[0]) & set(arrays[1])) or (set(arrays[0]) & set(arrays[2])) or (set(arrays[1]) & set(arrays[2])):
        raise RuntimeError("Numerical split overlap detected")
    return arrays


def arrays_from_dataframe(frame):
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame["Mental_Health_Status"].map(STATUS_TO_INDEX).to_numpy(dtype=np.int64)
    regression = frame[REGRESSION_TARGETS].to_numpy(dtype=np.float32)
    return x, y, regression
