import json

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_and_metrics():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["selected_models"]["face"] == "ConvNeXt-Tiny real-only"
    assert health.json()["selected_models"]["numerical"].startswith("synthetic-only")
    metrics = client.get("/api/model-metrics")
    assert metrics.status_code == 200
    assert metrics.json()["face"]["best_validation_macro_f1"] == 0.6799339209


def test_live_numerical_assessment_is_not_demo():
    values = {
        "Sleep_Quality": 3, "Social_Engagement": 3, "Daily_App_Usage_Min": 210,
        "Typing_Speed_WPM": 48, "Session_Frequency": 11, "Idle_Time_Min": 85,
        "Facial_Emotion_Variance": 0.52, "Eye_Blink_Rate": 19, "Smile_Intensity": 0.44,
        "Head_Motion_Index": 0.41, "MFCC_Mean": -8, "MFCC_Variance": 6.2,
        "Pitch_Mean": 188, "Speech_Rate": 3.6, "Heart_Rate_BPM": 80,
        "HRV_Index": 52, "Skin_Temperature": 33.6, "GSR_Level": 2.1,
    }
    response = TestClient(app).post("/api/assess", data={"numerical_json": json.dumps(values)})
    assert response.status_code == 200
    payload = response.json()
    assert "isDemo" not in payload
    assert payload["modality_weights"]["numerical"] == 1.0


def test_frontend_dev_origin_is_allowed():
    response = TestClient(app).options(
        "/api/assess",
        headers={
            "Origin": "http://127.0.0.1:5174",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"
