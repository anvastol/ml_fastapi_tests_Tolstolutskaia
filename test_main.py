from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "World"}


def test_predict_positive():
    response = client.post("/predict/",
                           json={"text": "I like machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_predict_negative():
    response = client.post("/predict/",
                           json={"text": "I hate machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'

def test_predict_invalid_input():
    response = client.post("/predict/", json={"invalid_key": "Invalid text"})
    assert response.status_code == 422
    # Проверяем, что код ответа - 422 (непроцессированная сущность)
    assert response.json() == {"detail": [{"loc": ["body", "text"],
                                           "msg": "field required",
                                           "type": "value_error.missing"}]}