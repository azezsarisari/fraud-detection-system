import time
import numpy as np
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
    confusion_matrix
)


DATA_PATH = "data/raw/paysim.csv"
CUTOFF_STEP = 378
THRESHOLD = 0.50


print("Loading PaySim dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df):,}")


# =========================================================
# TEMPORAL SPLIT
# =========================================================

train_df = df[
    df["step"] <= CUTOFF_STEP
].copy()

test_df = df[
    df["step"] > CUTOFF_STEP
].copy()


print("\n========================================")
print("TEMPORAL SPLIT")
print("========================================")

print(f"Train rows: {len(train_df):,}")
print(f"Train fraud: {train_df['isFraud'].sum():,}")

print(f"Test rows: {len(test_df):,}")
print(f"Test fraud: {test_df['isFraud'].sum():,}")


# =========================================================
# SAFE PRE-TRANSACTION FEATURE
# =========================================================

for data in [train_df, test_df]:

    data["amount_to_orig_balance"] = np.where(
        data["oldbalanceOrg"] > 0,
        data["amount"] / data["oldbalanceOrg"],
        0.0
    )


# =========================================================
# EXPERIMENTS
# =========================================================

experiments = {

    "A_type_only": [
        "type"
    ],

    "B_type_amount": [
        "type",
        "amount"
    ],

    "C_add_oldbalanceOrg": [
        "type",
        "amount",
        "oldbalanceOrg"
    ],

    "D_add_oldbalanceDest": [
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest"
    ],

    "E_add_ratio": [
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_orig_balance"
    ],

    "F_add_step": [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_orig_balance"
    ],

    "G_no_step_no_ratio": [
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest"
    ],

    "H_amount_ratio_only": [
        "amount",
        "oldbalanceOrg",
        "amount_to_orig_balance"
    ]
}


results = []


# =========================================================
# RUN EXPERIMENTS
# =========================================================

for name, features in experiments.items():

    print("\n")
    print("=" * 60)
    print(f"EXPERIMENT: {name}")
    print("=" * 60)

    print("Features:")
    for feature in features:
        print(f"  - {feature}")


    X_train = train_df[features].copy()
    y_train = train_df["isFraud"]

    X_test = test_df[features].copy()
    y_test = test_df["isFraud"]


    categorical_features = [
        col
        for col in features
        if col == "type"
    ]

    numeric_features = [
        col
        for col in features
        if col != "type"
    ]


    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        )


    preprocessor = ColumnTransformer(
        transformers=transformers
    )


    model = RandomForestClassifier(
        n_estimators=100,
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
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


    print("\nTraining...")

    start = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start
    )


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


    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()


    print(
        f"Training time: "
        f"{training_time / 60:.2f} min"
    )

    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")
    print(f"PR-AUC:    {pr_auc:.6f}")

    print(f"TP: {tp:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TN: {tn:,}")


    results.append({

        "experiment": name,
        "features": ", ".join(features),

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,

        "training_minutes":
            training_time / 60
    })


# =========================================================
# FINAL COMPARISON
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "f1",
    ascending=False
)


print("\n")
print("=" * 90)
print("ABLATION STUDY FINAL RESULTS")
print("=" * 90)

print(
    results_df[
        [
            "experiment",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_positives",
            "false_negatives"
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# SAVE RESULTS
# =========================================================

output_path = (
    "models/ablation_study_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nResults saved to: "
    f"{output_path}"
)

print(
    "\nAblation study completed successfully."
)
