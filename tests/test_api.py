from fastapi.testclient import TestClient

from src.api.main import app


def test_health_and_metrics():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["selected_models"]["face"] == "ConvNeXt-Tiny real-only"
    metrics = client.get("/api/model-metrics")
    assert metrics.status_code == 200
    assert metrics.json()["face"]["best_validation_macro_f1"] == 0.6799339209
