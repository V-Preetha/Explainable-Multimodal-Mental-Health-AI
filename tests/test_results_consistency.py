import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_results_match_raw_evidence():
    final = json.loads((ROOT / "results/final_results.json").read_text(encoding="utf-8"))
    face = json.loads((ROOT / "results/face/test_metrics.json").read_text(encoding="utf-8"))
    speech = json.loads((ROOT / "results/speech/final_metrics.json").read_text(encoding="utf-8"))
    fusion = json.loads((ROOT / "results/fusion/final_multimodal_metrics.json").read_text(encoding="utf-8"))["selected"]
    assert abs(face["test_accuracy"] - final["face"]["test_accuracy"]) < 1e-9
    assert abs(face["test_macro_f1"] - final["face"]["test_macro_f1"]) < 1e-9
    assert abs(speech["accuracy"] - final["speech_primary"]["accuracy"]) < 1e-9
    assert abs(speech["f1_macro"] - final["speech_primary"]["macro_f1"]) < 1e-9
    assert abs(fusion["test"]["accuracy"] - final["fusion"]["test_accuracy"]) < 1e-9
    assert abs(fusion["test"]["f1_macro"] - final["fusion"]["test_macro_f1"]) < 1e-9


def test_protocol_boundaries_are_explicit():
    final = json.loads((ROOT / "results/final_results.json").read_text(encoding="utf-8"))
    assert "not speaker-independent" in final["speech_random_split_ablation"]["protocol"]
    assert "not participant-paired" in final["fusion"]["protocol"]
    assert "synthetic held-out" in final["numerical_synthetic_only"]["protocol"].lower()
