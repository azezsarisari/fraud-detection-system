from typing import Literal
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    step: int = Field(..., ge=0)

    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    amount: float = Field(..., ge=0)

    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)

    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: str
    fraud_probability: float
    threshold: float
    risk_level: str


class BatchTransactionResult(BaseModel):
    transaction_id: int
    prediction: str
    fraud_probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    total_transactions: int
    fraud_detected: int
    legitimate_transactions: int
    critical_risk: int
    high_risk: int
    medium_risk: int
    low_risk: int
    threshold: float
    results: list[BatchTransactionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model: str
    dataset: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
