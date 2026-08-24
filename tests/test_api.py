from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


FRAUD_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}


LEGITIMATE_TRANSACTION = {
    "step": 1,
    "type": "PAYMENT",
    "amount": 9839.64,
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Fraud Detection API"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_loaded"] is True


def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "RandomForestClassifier"

    assert data["dataset"] == "PaySim"

    assert data["threshold"] == 0.5


def test_fraud_prediction():

    response = client.post(
        "/predict",
        json=FRAUD_TRANSACTION
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "FRAUD"

    assert data["fraud_probability"] >= 0.5

    assert data["risk_level"] == "CRITICAL"


def test_legitimate_prediction():

    response = client.post(
        "/predict",
        json=LEGITIMATE_TRANSACTION
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "LEGITIMATE"

    assert data["fraud_probability"] < 0.5


def test_batch_prediction():

    response = client.post(
        "/predict-batch",
        json=[
            FRAUD_TRANSACTION,
            LEGITIMATE_TRANSACTION
        ]
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_transactions"] == 2

    assert data["fraud_detected"] == 1

    assert data["legitimate_transactions"] == 1

    assert len(data["results"]) == 2


def test_invalid_transaction_type():

    invalid_transaction = (
        FRAUD_TRANSACTION.copy()
    )

    invalid_transaction["type"] = "INVALID"

    response = client.post(
        "/predict",
        json=invalid_transaction
    )

    assert response.status_code == 422


def test_negative_amount_rejected():

    invalid_transaction = (
        FRAUD_TRANSACTION.copy()
    )

    invalid_transaction["amount"] = -100

    response = client.post(
        "/predict",
        json=invalid_transaction
    )

    assert response.status_code == 422


def test_empty_batch_rejected():

    response = client.post(
        "/predict-batch",
        json=[]
    )

    assert response.status_code == 400
