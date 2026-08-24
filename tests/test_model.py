import joblib
import pandas as pd

from pathlib import Path


MODEL_PATH = Path(
    "models/fraud_detection_pipeline.joblib"
)


def test_model_file_exists():

    assert MODEL_PATH.exists()


def test_model_loads():

    model = joblib.load(
        MODEL_PATH
    )

    assert model is not None


def test_model_predicts_probability():

    model = joblib.load(
        MODEL_PATH
    )

    transaction = pd.DataFrame([
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
    ])

    probability = model.predict_proba(
        transaction
    )[0, 1]

    assert 0.0 <= probability <= 1.0

    assert probability >= 0.5
