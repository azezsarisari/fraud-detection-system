import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve
)


VAL_PATH = Path("data/processed/validation.csv")

MODEL_DIR = Path("models")
IMAGE_DIR = Path("images")

IMAGE_DIR.mkdir(exist_ok=True)

TARGET = "isFraud"


# =========================================================
# 1. LOAD VALIDATION DATA
# =========================================================

print("Loading validation data...")

val_df = pd.read_csv(VAL_PATH)

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

print(f"Validation rows: {len(val_df):,}")
print(f"Fraud cases: {y_val.sum():,}")


# =========================================================
# 2. MODELS
# =========================================================

models = {
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost_baseline.joblib"
}


results = []

probabilities = {}


# =========================================================
# 3. EVALUATE MODELS
# =========================================================

for model_name, model_path in models.items():

    print("\n========================================")
    print(f"EVALUATING: {model_name}")
    print("========================================")

    print(f"Loading: {model_path}")

    model = joblib.load(model_path)

    y_probability = model.predict_proba(X_val)[:, 1]

    y_pred = (
        y_probability >= 0.50
    ).astype(int)

    probabilities[model_name] = y_probability


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
        "model": model_name,
        "threshold": 0.50,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    })


    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")
    print(f"PR-AUC:    {pr_auc:.6f}")

    print(f"\nTP: {tp:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TN: {tn:,}")


    # =====================================================
    # CONFUSION MATRIX IMAGE
    # =====================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate",
            "Fraud"
        ]
    )

    display.plot(
        values_format=",d"
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        IMAGE_DIR
        / f"{filename}_confusion_matrix.png",
        dpi=150
    )

    plt.close()


# =========================================================
# 4. RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n========================================")
print("FINAL VALIDATION COMPARISON")
print("========================================")

print(
    results_df[
        [
            "model",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_positives",
            "false_negatives"
        ]
    ].to_string(index=False)
)


results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)


# =========================================================
# 5. PRECISION-RECALL CURVE
# =========================================================

plt.figure(figsize=(8, 6))

for model_name, y_probability in probabilities.items():

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_val,
            y_probability
        )
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=model_name
    )


plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "precision_recall_curve.png",
    dpi=150
)

plt.close()


# =========================================================
# 6. ROC CURVE
# =========================================================

plt.figure(figsize=(8, 6))

for model_name, y_probability in probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_val,
        y_probability
    )

    plt.plot(
        fpr,
        tpr,
        label=model_name
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "roc_curve.png",
    dpi=150
)

plt.close()


# =========================================================
# 7. SAVE VALIDATION PROBABILITIES
# =========================================================

probability_df = pd.DataFrame({
    "actual": y_val.to_numpy()
})

for model_name, y_probability in probabilities.items():

    column_name = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_probability"
    )

    probability_df[column_name] = y_probability


probability_df.to_csv(
    MODEL_DIR / "validation_probabilities.csv",
    index=False
)


print("\nSaved:")
print("models/model_comparison.csv")
print("models/validation_probabilities.csv")

print("\nCharts saved:")
print("images/random_forest_confusion_matrix.png")
print("images/xgboost_confusion_matrix.png")
print("images/precision_recall_curve.png")
print("images/roc_curve.png")

print(
    "\nPhase 9 model evaluation completed successfully."
)
