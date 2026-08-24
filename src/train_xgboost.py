import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "isFraud"


print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"Training rows: {len(train_df):,}")

print("\nLoading validation data...")
val_df = pd.read_csv(VAL_PATH)
print(f"Validation rows: {len(val_df):,}")


X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]


categorical_features = ["type"]

numeric_features = [
    col
    for col in X_train.columns
    if col not in categorical_features
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


model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


print("\nTraining XGBoost...")
print("This may take several minutes.")

start_time = time.time()

pipeline.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("Training completed.")
print(
    f"Training time: {training_time / 60:.2f} minutes"
)


print("\nRunning validation predictions...")

y_pred = pipeline.predict(X_val)

y_probability = pipeline.predict_proba(
    X_val
)[:, 1]


precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    y_probability
)

pr_auc = average_precision_score(
    y_val,
    y_probability
)


cm = confusion_matrix(
    y_val,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


print("\n================================")
print("XGBOOST VALIDATION RESULTS")
print("================================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\n=== CONFUSION MATRIX ===")
print(cm)

print("\n=== FRAUD DETECTION DETAILS ===")
print(f"True Positives:  {tp:,}")
print(f"False Negatives: {fn:,}")
print(f"False Positives: {fp:,}")
print(f"True Negatives:  {tn:,}")

print("\n=== CLASSIFICATION REPORT ===")

print(
    classification_report(
        y_val,
        y_pred,
        digits=4,
        zero_division=0
    )
)


print("\n================================")
print("MODEL COMPARISON")
print("================================")

print("\nLogistic Regression:")
print("Precision: 0.9286")
print("Recall:    0.5698")
print("F1:        0.7062")
print("ROC-AUC:   0.9962")
print("PR-AUC:    0.7975")
print("Missed Fraud: 530")

print("\nRandom Forest:")
print("Precision: 1.0000")
print("Recall:    0.9968")
print("F1:        0.9984")
print("ROC-AUC:   0.9996")
print("PR-AUC:    0.9979")
print("Missed Fraud: 4")

print("\nXGBoost:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"Missed Fraud: {fn:,}")


MODEL_PATH = (
    MODEL_DIR
    / "xgboost_baseline.joblib"
)

print("\nSaving XGBoost model...")

joblib.dump(
    pipeline,
    MODEL_PATH,
    compress=3
)

print(
    f"Model saved to: {MODEL_PATH}"
)

print(
    "\nXGBoost training completed successfully."
)
