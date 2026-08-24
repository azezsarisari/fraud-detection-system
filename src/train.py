import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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


# =========================================================
# 1. LOAD DATA
# =========================================================

print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)

print(f"Training rows: {len(train_df):,}")


print("\nLoading validation data...")

val_df = pd.read_csv(VAL_PATH)

print(f"Validation rows: {len(val_df):,}")


# =========================================================
# 2. SPLIT FEATURES AND TARGET
# =========================================================

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]


print("\nTraining features:")
print(X_train.columns.tolist())


# =========================================================
# 3. DEFINE FEATURE TYPES
# =========================================================

categorical_features = [
    "type"
]

numeric_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)


# =========================================================
# 4. PREPROCESSING PIPELINE
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# =========================================================
# 5. BASELINE MODEL
# =========================================================

model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    random_state=42
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# =========================================================
# 6. TRAIN
# =========================================================

print("\nTraining Logistic Regression baseline...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# =========================================================
# 7. VALIDATION PREDICTIONS
# =========================================================

print("\nRunning validation predictions...")

y_pred = pipeline.predict(X_val)

y_probability = pipeline.predict_proba(
    X_val
)[:, 1]


# =========================================================
# 8. EVALUATION
# =========================================================

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


print("\n================================")
print("BASELINE VALIDATION RESULTS")
print("================================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")


print("\n=== CONFUSION MATRIX ===")

cm = confusion_matrix(
    y_val,
    y_pred
)

print(cm)


print("\n=== CLASSIFICATION REPORT ===")

print(
    classification_report(
        y_val,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 9. SAVE BASELINE MODEL
# =========================================================

MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression_baseline.joblib"
)

joblib.dump(
    pipeline,
    MODEL_PATH
)


print(
    f"\nBaseline model saved to: {MODEL_PATH}"
)

print(
    "\nPhase 6 baseline training completed successfully."
)
