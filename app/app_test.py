import pytest
from fastapi.testclient import TestClient

from app.app import app


@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)


def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health_check": "OK"}


def test_model(test_client):
    objects = [
        {
            "features": {
                "feature_19": 1.0,
                "feature_23": 0.0,
                "feature_66": 0.77,
                "feature_55": 0.75,
                "feature_21": 1,
                "feature_13": 0.7,
                "feature_9": 6,
                "feature_4": 40,
                "feature_20": 200,
                "feature_7": 10,
                "feature_11": 120,
                "feature_22": 40,
                "feature_16": 10,
                "feature_10": 1,
            }
        },
        {
            "features": {
                "feature_19": 1.0,
                "feature_23": 0.0,
                "feature_66": 0.77,
                "feature_55": 0.75,
                "feature_21": 1,
                "feature_13": 0.7,
                "feature_9": 6,
                "feature_4": 40,
                "feature_20": 200,
                "feature_7": 10,
                "feature_11": 120,
                "feature_22": 40,
                "feature_16": 10,
                "feature_10": 1,
            }
        },
    ]

    response = test_client.post("/prediction", json=objects)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == len(objects)
