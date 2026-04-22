from fastapi.testclient import TestClient

from src import api


def test_predict_requires_loaded_model(): #The API should reject inference before setup.
    api.model = None
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", b"fake-image-bytes", "image/png")},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}