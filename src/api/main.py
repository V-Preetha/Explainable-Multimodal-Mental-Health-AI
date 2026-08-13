from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from src.face.model import ConvNeXtEmotion
from src.fusion.model import GatedFusionMultiTask
from src.numerical.common import FEATURES, STATUS_CLASSES
from src.numerical.models import NumericalMultiTask
from src.speech.features import get_encoder
from src.speech.models import SpeechEmotionMLP


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SCORES = np.asarray([34.0, 24.0, 39.0], dtype=np.float32)
DISCLAIMER = "Research decision-support prototype only; this output is not a clinical diagnosis."

FACE_STRESS = {"Happy": 0, "Neutral": 0, "Sad": 1, "Surprise": 1, "Fear": 2, "Disgust": 2, "Angry": 3}
SPEECH_STRESS = {"happy": 0, "neutral": 0, "calm": 0, "sad": 1, "surprised": 1, "fearful": 2, "angry": 2, "disgust": 3}


def _checkpoint(name: str) -> dict:
    path = MODEL_DIR / name
    if not path.exists():
        raise RuntimeError(f"Missing selected model checkpoint: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


face_checkpoint = _checkpoint("convnext_face_winner.pt")
face_model = ConvNeXtEmotion(pretrained=False).to(DEVICE)
face_model.load_state_dict(face_checkpoint["state_dict"])
face_model.eval()
FACE_CLASSES = face_checkpoint["classes"]
FACE_SIZE = int(face_checkpoint.get("image_size", 224))
face_transform = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

speech_checkpoint = _checkpoint("speech_emotion2vec_strict.pt")
speech_model = SpeechEmotionMLP(
    input_dim=int(speech_checkpoint.get("input_dim", 768)),
    embedding_dim=int(speech_checkpoint.get("embedding_dim", 256)),
    classes=len(speech_checkpoint["classes"]),
).to(DEVICE)
speech_model.load_state_dict(speech_checkpoint["state_dict"])
speech_model.eval()
SPEECH_CLASSES = speech_checkpoint["classes"]

numerical_checkpoint = _checkpoint("numerical_multitask.pt")
numerical_model = NumericalMultiTask().to(DEVICE)
numerical_model.load_state_dict(numerical_checkpoint["state_dict"])
numerical_model.eval()
numerical_scaler = joblib.load(MODEL_DIR / "numerical_scaler.joblib")

fusion_checkpoint = _checkpoint("fusion_all_modalities.pt")
fusion_model = GatedFusionMultiTask().to(DEVICE)
fusion_model.load_state_dict(fusion_checkpoint["state_dict"])
fusion_model.eval()

app = FastAPI(title="Aegis Multimodal Research API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _confidence(probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities.astype(float), 1e-12, 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.clip(1.0 - entropy / math.log(len(probabilities)), 0.0, 1.0))


def _stress_evidence(probabilities: dict[str, float], mapping: dict[str, int]) -> np.ndarray:
    evidence = np.zeros(4, dtype=np.float32)
    for label, value in probabilities.items():
        evidence[mapping[label]] += value
    return evidence / max(float(evidence.sum()), 1e-8)


def _agreement(distributions: list[np.ndarray]) -> float:
    if len(distributions) < 2:
        return 1.0
    distances = []
    for left_index in range(len(distributions)):
        for right_index in range(left_index + 1, len(distributions)):
            left = np.clip(distributions[left_index], 1e-12, 1.0)
            right = np.clip(distributions[right_index], 1e-12, 1.0)
            midpoint = (left + right) / 2
            divergence = 0.5 * np.sum(left * np.log2(left / midpoint)) + 0.5 * np.sum(right * np.log2(right / midpoint))
            distances.append(divergence)
    return float(np.clip(1.0 - np.mean(distances), 0.0, 1.0))


def _face_inference(raw: bytes):
    try:
        image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise ValueError("Invalid or corrupted face image.") from exc
    if image.width < 16 or image.height < 16 or np.asarray(image).std() < 1:
        raise ValueError("The face image is blank or contains too little visual information.")
    tensor = face_transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        embedding, logits = face_model(tensor, return_embedding=True)
        probabilities = logits.softmax(dim=1)[0].cpu().numpy()
    values = dict(zip(FACE_CLASSES, map(float, probabilities)))
    label = FACE_CLASSES[int(probabilities.argmax())]
    return embedding, values, label, _stress_evidence(values, FACE_STRESS)


def _speech_inference(raw: bytes):
    try:
        waveform, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
    except Exception as exc:
        raise ValueError("Invalid or corrupted audio file.") from exc
    if len(waveform) < sample_rate // 4 or float(np.std(waveform)) < 1e-5:
        raise ValueError("The audio is silent or too short.")
    if sample_rate != 16000:
        output_length = int(len(waveform) * 16000 / sample_rate)
        waveform = np.interp(np.linspace(0, len(waveform) - 1, output_length), np.arange(len(waveform)), waveform).astype("float32")
    encoder = get_encoder(str(DEVICE))
    result = encoder.generate(input=waveform, granularity="frame", extract_embedding=True)[0]
    frames = torch.as_tensor(result["feats"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=DEVICE)
    with torch.inference_mode():
        embedding, logits, attention = speech_model(frames, mask)
        probabilities = logits.softmax(dim=1)[0].cpu().numpy()
    values = dict(zip(SPEECH_CLASSES, map(float, probabilities)))
    label = SPEECH_CLASSES[int(probabilities.argmax())]
    return embedding, values, label, attention[0].cpu().tolist(), _stress_evidence(values, SPEECH_STRESS)


def _numerical_inference(values: dict[str, float]):
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError("All 18 numerical indicators are required when numerical input is used. Missing: " + ", ".join(missing))
    raw = np.asarray([[values[feature] for feature in FEATURES]], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Numerical indicators must be finite values.")
    scaled = numerical_scaler.transform(raw).astype(np.float32)
    tensor = torch.from_numpy(scaled).to(DEVICE)
    with torch.inference_mode():
        embedding, logits, _ = numerical_model(tensor)
        probabilities = logits.softmax(dim=1)[0].cpu().numpy()
    return embedding, dict(zip(STATUS_CLASSES, map(float, probabilities))), probabilities


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "selected_models": {
            "face": "ConvNeXt-Tiny real-only",
            "speech": "emotion2vec+ actor-independent",
            "numerical": "numerical multi-task encoder",
            "fusion": "gated all-modalities fusion",
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/model-metrics")
def model_metrics():
    return json.loads((ROOT / "results" / "final_results.json").read_text(encoding="utf-8"))


@app.post("/api/assess")
async def assess(
    face_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    numerical_json: Optional[str] = Form(None),
):
    face_embedding = torch.zeros((1, 256), device=DEVICE)
    speech_embedding = torch.zeros((1, 256), device=DEVICE)
    numerical_embedding = torch.zeros((1, 128), device=DEVICE)
    available = [False, False, False]
    modalities: dict = {}
    explainability: dict = {}
    evidence: list[np.ndarray] = []

    try:
        if face_file is not None:
            face_embedding, probabilities, label, distribution = _face_inference(await face_file.read())
            available[0] = True
            modalities["face"] = {"emotion": label, "confidence": max(probabilities.values())}
            explainability["face"] = {"predicted_emotion": label, "confidence": max(probabilities.values())}
            evidence.append(distribution)
        if audio_file is not None:
            speech_embedding, probabilities, label, attention, distribution = _speech_inference(await audio_file.read())
            available[1] = True
            modalities["speech"] = {"emotion": label, "confidence": max(probabilities.values())}
            explainability["speech"] = {"predicted_emotion": label, "confidence": max(probabilities.values()), "temporal_attention": attention}
            evidence.append(distribution)
        if numerical_json is not None:
            try:
                numerical_values = json.loads(numerical_json)
            except json.JSONDecodeError as exc:
                raise ValueError("Numerical input is not valid JSON.") from exc
            numerical_embedding, probabilities, distribution = _numerical_inference(numerical_values)
            available[2] = True
            modalities["numerical"] = {"confidence": max(probabilities.values())}
            evidence.append(distribution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not any(available):
        raise HTTPException(status_code=422, detail="At least one valid modality is required.")

    availability = torch.tensor([available], dtype=torch.bool, device=DEVICE)
    with torch.inference_mode():
        status_logits, score_values, weights, _ = fusion_model(
            face_embedding, speech_embedding, numerical_embedding, availability
        )
        status_probabilities = status_logits.softmax(dim=1)[0].cpu().numpy()
        scores = score_values[0].cpu().numpy() * MAX_SCORES
        modality_weights = weights[0].cpu().numpy()

    agreement = _agreement(evidence)
    raw_confidence = _confidence(status_probabilities)
    confidence = raw_confidence * agreement
    return {
        "status": STATUS_CLASSES[int(status_probabilities.argmax())],
        "confidence": confidence,
        "scores": {"depression": float(scores[0]), "anxiety": float(scores[1]), "stress": float(scores[2])},
        "modalities": modalities,
        "modality_weights": {
            "face": float(modality_weights[0]),
            "speech": float(modality_weights[1]),
            "numerical": float(modality_weights[2]),
        },
        "explainability": explainability,
        "reliability": {
            "overall_confidence": confidence,
            "modality_agreement": agreement,
            "mixed_evidence": agreement < 0.5,
        },
        "disclaimer": DISCLAIMER,
    }
