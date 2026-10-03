from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained["artifacts"]))
    from diabetes_xai.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from diabetes_xai.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_health_without_model(client_no_model):
    assert client_no_model.get("/health").status_code == 503
    assert client_no_model.post("/predict", json={}).status_code in (422, 503)


def test_predict_contract(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert {"predicted_class", "label", "probability", "threshold", "model_version"} <= set(body)
    assert body["top_factors"] is None
    assert "x-request-id" in r.headers


def test_predict_with_explanation(client, patient):
    body = client.post("/predict?explain=true", json=patient).json()
    assert len(body["top_factors"]) == 5


@pytest.mark.parametrize(
    "field,value",
    [("BMI", 5), ("GenHlth", 6), ("Age", 0), ("HighBP", 2), ("MentHlth", 31), ("Income", "high")],
)
def test_validation_errors(client, patient, field, value):
    r = client.post("/predict", json={**patient, field: value})
    assert r.status_code == 422
    assert str(value) not in r.text or field in r.text


def test_rejects_unknown_and_missing_fields(client, patient):
    assert client.post("/predict", json={**patient, "name": "x"}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in patient.items() if k != "Age"}).status_code == 422


def test_model_info(client):
    info = client.get("/model/info").json()
    assert len(info["features"]) == 21 and 0 < info["threshold"] < 1
