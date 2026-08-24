import pandas as pd
import numpy as np
import joblib
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
    confusion_matrix
)


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")

MODEL_PATH = Path("models/random_forest.joblib")

TARGET = "isFraud"


print("========================================")
print("PART 1 - ENGINEERED FEATURE IMPORTANCE")
print("========================================")

print("\nLoading trained Random Forest...")

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(
    "importance",
    ascending=False
)

print("\nTop feature importances:")
print(
    importance_df.head(20).to_string(index=False)
)


print("\n========================================")
print("PART 2 - RAW FEATURES RANDOM FOREST")
print("========================================")

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

raw_features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X_train = train_df[raw_features]
y_train = train_df[TARGET]

X_val = val_df[raw_features]
y_val = val_df[TARGET]


categorical_features = ["type"]

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]


preprocessor_raw = ColumnTransformer(
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


raw_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)


raw_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_raw),
        ("model", raw_model)
    ]
)


print("\nTraining Random Forest with RAW features only...")

raw_pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


print("\nRunning validation predictions...")

y_pred = raw_pipeline.predict(X_val)

y_probability = raw_pipeline.predict_proba(
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


print("\n========================================")
print("RAW FEATURES RANDOM FOREST RESULTS")
print("========================================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nFraud Detection:")
print(f"True Positives:  {tp:,}")
print(f"False Negatives: {fn:,}")
print(f"False Positives: {fp:,}")
print(f"True Negatives:  {tn:,}")


print("\n========================================")
print("ENGINEERED VS RAW")
print("========================================")

print("\nEngineered Features RF:")
print("Precision: 1.0000")
print("Recall:    0.9968")
print("F1:        0.9984")
print("ROC-AUC:   0.9996")
print("PR-AUC:    0.9979")
print("Missed Fraud: 4")

print("\nRaw Features RF:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"Missed Fraud: {fn:,}")


print("\nSanity check completed successfully.")
