import pandas as pd
import joblib
import json
import time

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.transformers import FraudFeatureEngineer


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "isFraud"
THRESHOLD = 0.50


print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"Training rows: {len(train_df):,}")

print("\nLoading FINAL test data...")
test_df = pd.read_csv(TEST_PATH)
print(f"Test rows: {len(test_df):,}")
print(f"Test fraud cases: {test_df[TARGET].sum():,}")


raw_features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]


X_train = train_df[raw_features].copy()
y_train = train_df[TARGET]

X_test = test_df[raw_features].copy()
y_test = test_df[TARGET]


numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_change",
    "dest_balance_change",
    "orig_balance_error",
    "dest_balance_error",
    "amount_to_orig_balance",
    "orig_account_emptied"
]

categorical_features = [
    "type"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


classifier = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)


production_pipeline = Pipeline(
    steps=[
        (
            "feature_engineering",
            FraudFeatureEngineer()
        ),
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            classifier
        )
    ]
)


print("\nTraining production pipeline...")

start_time = time.time()

production_pipeline.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("Training completed.")
print(
    f"Training time: {training_time / 60:.2f} minutes"
)


MODEL_PATH = (
    MODEL_DIR
    / "fraud_detection_pipeline.joblib"
)

joblib.dump(
    production_pipeline,
    MODEL_PATH,
    compress=3
)

print(f"\nSaved model to: {MODEL_PATH}")


print("\nRunning final test...")

y_probability = production_pipeline.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_probability >= THRESHOLD
).astype(int)


precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


print("\nFINAL TEST RESULTS")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")

print("\nConfusion Matrix:")
print(cm)

print(
    classification_report(
        y_test,
        y_pred,
        digits=6,
        zero_division=0
    )
)


metadata = {
    "model": "RandomForestClassifier",
    "dataset": "PaySim",
    "threshold": THRESHOLD,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_negatives": int(tn)
}


with open(
    MODEL_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print("\nProduction model rebuilt successfully.")
