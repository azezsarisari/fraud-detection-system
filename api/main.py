import json
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException

from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)


MODEL_PATH = Path(
    "models/fraud_detection_pipeline.joblib"
)

METADATA_PATH = Path(
    "models/metadata.json"
)


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Machine learning API for detecting "
        "fraudulent financial transactions."
    ),
    version="1.0.0"
)


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = joblib.load(MODEL_PATH)

except Exception as error:
    model = None
    print(
        f"Failed to load model: {error}"
    )


# =========================================================
# LOAD METADATA
# =========================================================

try:
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

except Exception as error:
    metadata = {}
    print(
        f"Failed to load metadata: {error}"
    )


THRESHOLD = metadata.get(
    "threshold",
    0.50
)


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(
    probability: float
) -> str:

    if probability >= 0.90:
        return "CRITICAL"

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Fraud Detection API",
        "docs": "/docs"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    return {
        "status": (
            "healthy"
            if model is not None
            else "unhealthy"
        ),
        "model_loaded": (
            model is not None
        )
    }


# =========================================================
# MODEL INFO
# =========================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():

    if not metadata:

        raise HTTPException(
            status_code=500,
            detail="Model metadata not available."
        )

    return {
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "threshold": metadata["threshold"],
        "precision": metadata["precision"],
        "recall": metadata["recall"],
        "f1": metadata["f1"],
        "roc_auc": metadata["roc_auc"],
        "pr_auc": metadata["pr_auc"]
    }


# =========================================================
# SINGLE PREDICTION
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    transaction: TransactionInput
):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    input_data = pd.DataFrame(
        [
            transaction.model_dump()
        ]
    )

    try:

        probability = float(
            model.predict_proba(
                input_data
            )[0, 1]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )

    prediction = (
        "FRAUD"
        if probability >= THRESHOLD
        else "LEGITIMATE"
    )

    return {
        "prediction": prediction,
        "fraud_probability": probability,
        "threshold": THRESHOLD,
        "risk_level": get_risk_level(
            probability
        )
    }


# =========================================================
# BATCH PREDICTION
# =========================================================

@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse
)
def predict_batch(
    transactions: list[TransactionInput]
):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    if len(transactions) == 0:

        raise HTTPException(
            status_code=400,
            detail="No transactions provided."
        )

    if len(transactions) > 10000:

        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum batch size is "
                "10,000 transactions."
            )
        )

    input_data = pd.DataFrame(
        [
            transaction.model_dump()
            for transaction in transactions
        ]
    )

    try:

        probabilities = (
            model.predict_proba(
                input_data
            )[:, 1]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Batch prediction failed: "
                f"{error}"
            )
        )

    results = []

    fraud_count = 0
    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    for index, probability in enumerate(
        probabilities
    ):

        probability = float(probability)

        prediction = (
            "FRAUD"
            if probability >= THRESHOLD
            else "LEGITIMATE"
        )

        risk_level = get_risk_level(
            probability
        )

        if prediction == "FRAUD":
            fraud_count += 1

        if risk_level == "CRITICAL":
            critical_count += 1

        elif risk_level == "HIGH":
            high_count += 1

        elif risk_level == "MEDIUM":
            medium_count += 1

        else:
            low_count += 1

        results.append({
            "transaction_id": index + 1,
            "prediction": prediction,
            "fraud_probability": probability,
            "risk_level": risk_level
        })

    total = len(transactions)

    return {
        "total_transactions": total,
        "fraud_detected": fraud_count,
        "legitimate_transactions": (
            total - fraud_count
        ),
        "critical_risk": critical_count,
        "high_risk": high_count,
        "medium_risk": medium_count,
        "low_risk": low_count,
        "threshold": THRESHOLD,
        "results": results
    }
