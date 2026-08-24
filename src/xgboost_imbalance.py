import pandas as pd
import time
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

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

TARGET = "isFraud"


print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"Training rows:   {len(train_df):,}")
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


normal_count = (y_train == 0).sum()
fraud_count = (y_train == 1).sum()

natural_ratio = normal_count / fraud_count


print("\n=== CLASS DISTRIBUTION ===")

print(f"Normal: {normal_count:,}")
print(f"Fraud:  {fraud_count:,}")
print(
    f"Natural scale_pos_weight: "
    f"{natural_ratio:.2f}"
)


weights = [
    1,
    10,
    50,
    100,
    300,
    round(natural_ratio, 2)
]


results = []


for weight in weights:

    print("\n========================================")
    print(f"scale_pos_weight = {weight}")
    print("========================================")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=weight,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    start = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start

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


    results.append({
        "weight": weight,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "training_seconds": training_time
    })


    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")

    print(
        f"Training time: "
        f"{training_time / 60:.2f} minutes"
    )


results_df = pd.DataFrame(results)


print("\n========================================")
print("WEIGHT COMPARISON")
print("========================================")

print(
    results_df[
        [
            "weight",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "false_positives",
            "false_negatives"
        ]
    ].to_string(index=False)
)


results_df.to_csv(
    "models/xgboost_weight_experiments.csv",
    index=False
)


print(
    "\nResults saved to:"
    "\nmodels/xgboost_weight_experiments.csv"
)

print(
    "\nXGBoost imbalance experiment completed."
)
