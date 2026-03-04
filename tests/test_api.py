from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Footballer Value Predictor API is running!"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    payload = {
        "age": 22,
        "overall": 82,
        "potential": 91,
        "wage_eur": 25000,
        "international_reputation": 2,
        "weak_foot": 3,
        "skill_moves": 4,
        "pace": 85.0,
        "shooting": 78.0,
        "passing": 80.0,
        "dribbling": 86.0,
        "defending": 32.0,
        "physic": 68.0,
        "preferred_foot": 1,
        "years_on_contract": 3,
        "position_ATT": 1,
        "position_DEF": 0,
        "position_GK": 0,
        "position_MID": 0,
        "attack_work_rate_High": 1,
        "attack_work_rate_Low": 0,
        "attack_work_rate_Medium": 0,
        "defense_work_rate_High": 0,
        "defense_work_rate_Low": 0,
        "defense_work_rate_Medium": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_value_eur" in response.json()
    assert "predicted_value_millions" in response.json()
    assert response.json()["predicted_value_eur"] > 0