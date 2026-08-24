import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


PROB_PATH = Path("models/validation_probabilities.csv")
MODEL_DIR = Path("models")
IMAGE_DIR = Path("images")

IMAGE_DIR.mkdir(exist_ok=True)


print("Loading validation probabilities...")

df = pd.read_csv(PROB_PATH)

y_true = df["actual"].to_numpy()

print(f"Validation rows: {len(df):,}")
print(f"Fraud cases: {y_true.sum():,}")


models = {
    "Random Forest": "random_forest_probability",
    "XGBoost": "xgboost_probability"
}


# Fine threshold grid.
# Includes 0.50 explicitly.
thresholds = np.unique(
    np.concatenate([
        np.arange(0.01, 1.00, 0.01),
        np.array([0.50])
    ])
)


all_results = []


for model_name, probability_column in models.items():

    print("\n========================================")
    print(f"OPTIMIZING: {model_name}")
    print("========================================")

    y_probability = df[
        probability_column
    ].to_numpy()


    model_results = []


    for threshold in thresholds:

        y_pred = (
            y_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()


        row = {
            "model": model_name,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn
        }

        model_results.append(row)
        all_results.append(row)


    model_df = pd.DataFrame(
        model_results
    )


    # ==============================================
    # BEST F1
    # ==============================================

    best_f1_row = model_df.loc[
        model_df["f1"].idxmax()
    ]


    # ==============================================
    # HIGH RECALL WHILE KEEPING PRECISION >= 99%
    # ==============================================

    high_precision_df = model_df[
        model_df["precision"] >= 0.99
    ].copy()


    if not high_precision_df.empty:

        best_recall_row = (
            high_precision_df
            .sort_values(
                by=[
                    "recall",
                    "false_negatives",
                    "false_positives",
                    "f1"
                ],
                ascending=[
                    False,
                    True,
                    True,
                    False
                ]
            )
            .iloc[0]
        )

    else:
        best_recall_row = None


    # ==============================================
    # CURRENT THRESHOLD = 0.50
    # ==============================================

    current_row = model_df[
        np.isclose(
            model_df["threshold"],
            0.50
        )
    ].iloc[0]


    print("\n--- CURRENT THRESHOLD 0.50 ---")

    print(
        f"Precision: {current_row['precision']:.6f}"
    )

    print(
        f"Recall:    {current_row['recall']:.6f}"
    )

    print(
        f"F1:        {current_row['f1']:.6f}"
    )

    print(
        f"FP: {int(current_row['false_positives']):,}"
    )

    print(
        f"FN: {int(current_row['false_negatives']):,}"
    )


    print("\n--- BEST F1 THRESHOLD ---")

    print(
        f"Threshold: {best_f1_row['threshold']:.2f}"
    )

    print(
        f"Precision: {best_f1_row['precision']:.6f}"
    )

    print(
        f"Recall:    {best_f1_row['recall']:.6f}"
    )

    print(
        f"F1:        {best_f1_row['f1']:.6f}"
    )

    print(
        f"FP: {int(best_f1_row['false_positives']):,}"
    )

    print(
        f"FN: {int(best_f1_row['false_negatives']):,}"
    )


    if best_recall_row is not None:

        print(
            "\n--- BEST RECALL WITH PRECISION >= 99% ---"
        )

        print(
            f"Threshold: "
            f"{best_recall_row['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_recall_row['precision']:.6f}"
        )

        print(
            f"Recall:    "
            f"{best_recall_row['recall']:.6f}"
        )

        print(
            f"F1:        "
            f"{best_recall_row['f1']:.6f}"
        )

        print(
            f"FP: "
            f"{int(best_recall_row['false_positives']):,}"
        )

        print(
            f"FN: "
            f"{int(best_recall_row['false_negatives']):,}"
        )


    # ==============================================
    # PRECISION / RECALL / F1 VS THRESHOLD
    # ==============================================

    plt.figure(figsize=(9, 6))

    plt.plot(
        model_df["threshold"],
        model_df["precision"],
        label="Precision"
    )

    plt.plot(
        model_df["threshold"],
        model_df["recall"],
        label="Recall"
    )

    plt.plot(
        model_df["threshold"],
        model_df["f1"],
        label="F1"
    )

    plt.axvline(
        x=0.50,
        linestyle="--",
        label="Default 0.50"
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(
        f"{model_name} - Threshold Optimization"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        IMAGE_DIR
        / f"{filename}_threshold_optimization.png",
        dpi=150
    )

    plt.close()


# =========================================================
# SAVE ALL RESULTS
# =========================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    MODEL_DIR / "threshold_results.csv",
    index=False
)


# =========================================================
# GLOBAL BEST F1
# =========================================================

global_best = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n========================================")
print("GLOBAL BEST F1 RESULT")
print("========================================")

print(
    f"Model:     {global_best['model']}"
)

print(
    f"Threshold: {global_best['threshold']:.2f}"
)

print(
    f"Precision: {global_best['precision']:.6f}"
)

print(
    f"Recall:    {global_best['recall']:.6f}"
)

print(
    f"F1:        {global_best['f1']:.6f}"
)

print(
    f"FP: {int(global_best['false_positives']):,}"
)

print(
    f"FN: {int(global_best['false_negatives']):,}"
)


print("\nSaved:")
print("models/threshold_results.csv")

print("\nCharts saved:")
print(
    "images/random_forest_threshold_optimization.png"
)
print(
    "images/xgboost_threshold_optimization.png"
)

print(
    "\nPhase 10 threshold optimization completed successfully."
)
