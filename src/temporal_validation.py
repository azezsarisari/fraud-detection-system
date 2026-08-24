import time
import pandas as pd

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


DATA_PATH = "data/raw/paysim.csv"
CUTOFF_STEP = 378
THRESHOLD = 0.50

RAW_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "type"
]


print("Loading PaySim dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df):,}")


print("\nCreating temporal split...")

train_df = df[
    df["step"] <= CUTOFF_STEP
].copy()

test_df = df[
    df["step"] > CUTOFF_STEP
].copy()


print("\n========================================")
print("TEMPORAL SPLIT")
print("========================================")

print(f"Cutoff step: {CUTOFF_STEP}")

print(f"\nTrain rows: {len(train_df):,}")
print(f"Train fraud: {train_df['isFraud'].sum():,}")
print(
    f"Train fraud rate: "
    f"{train_df['isFraud'].mean() * 100:.4f}%"
)

print(f"\nTest rows: {len(test_df):,}")
print(f"Test fraud: {test_df['isFraud'].sum():,}")
print(
    f"Test fraud rate: "
    f"{test_df['isFraud'].mean() * 100:.4f}%"
)


X_train = train_df[RAW_FEATURES].copy()
y_train = train_df["isFraud"]

X_test = test_df[RAW_FEATURES].copy()
y_test = test_df["isFraud"]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES
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


pipeline = Pipeline(
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


print("\nTraining temporal Random Forest...")

start = time.time()

pipeline.fit(
    X_train,
    y_train
)

training_time = time.time() - start

print(
    f"Training completed in "
    f"{training_time / 60:.2f} minutes."
)


print("\nPredicting future transactions...")

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()


print("\n========================================")
print("TEMPORAL HOLDOUT RESULTS")
print("========================================")

print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nFraud Detection:")
print(f"True Positives:  {tp:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Negatives:  {tn:,}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=6,
        zero_division=0
    )
)


print("\n========================================")
print("CURRENT RANDOM HOLDOUT VS TEMPORAL")
print("========================================")

print("\nCurrent Random Holdout:")
print("Precision: 1.000000")
print("Recall:    0.995942")
print("F1:        0.997967")
print("ROC-AUC:   0.998692")
print("PR-AUC:    0.996020")
print("False Positives: 0")
print("False Negatives: 5")

print("\nTemporal Holdout:")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1:        {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")

print("\nTemporal validation completed.")
