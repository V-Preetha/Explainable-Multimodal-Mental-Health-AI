from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = (
    "Research decision-support prototype only. The fusion data are weakly class-conditionally "
    "aligned across independent datasets and are not participant-paired clinical evidence."
)

app = FastAPI(title="Aegis Multimodal Research API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"] , allow_headers=["*"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "selected_models": {
            "face": "ConvNeXt-Tiny real-only",
            "speech": "emotion2vec+ actor-independent",
            "numerical": "original baseline plus separately reported synthetic benchmarks",
            "fusion": "gated weak class-conditional fusion",
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/model-metrics")
def model_metrics():
    """Return the single authoritative machine-readable summary used by the demo."""
    return json.loads((ROOT / "results" / "final_results.json").read_text(encoding="utf-8"))

