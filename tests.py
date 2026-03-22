import json
import pytest
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    rv = client.get('/health')
    assert rv.status_code == 200
    assert b"Flask Churn Prediction API is running." in rv.data

def test_predict(client):
    data = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 50000.0
    }
    rv = client.post('/predict', json=data)
    assert rv.status_code == 200
    resp_data = json.loads(rv.data)
    assert 'churn_tahmini' in resp_data
    assert 'risk_seviyesi' in resp_data
    assert 'shap_degerleri' in resp_data

def test_predict_batch(client):
    data = [
        {"CustomerId": 1, "CreditScore": 600, "Geography": "France", "Gender": "Female", "Age": 40, "Tenure": 3, "Balance": 60000.0, "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 50000.0},
        {"CustomerId": 2, "CreditScore": 800, "Geography": "Germany", "Gender": "Male", "Age": 25, "Tenure": 1, "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 0, "IsActiveMember": 1, "EstimatedSalary": 150000.0}
    ]
    rv = client.post('/predict_batch', json=data)
    assert rv.status_code == 200
    resp_data = json.loads(rv.data)
    assert resp_data['status'] == 'success'
    assert len(resp_data['results']) == 2
