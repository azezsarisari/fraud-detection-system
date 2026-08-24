import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path


VAL_PATH = Path("data/processed/validation.csv")
MODEL_PATH = Path("models/random_forest.joblib")

IMAGE_DIR = Path("images")
MODEL_DIR = Path("models")

IMAGE_DIR.mkdir(exist_ok=True)

TARGET = "isFraud"

RANDOM_STATE = 42


print("Loading validation data...")

val_df = pd.read_csv(VAL_PATH)

print(f"Validation rows: {len(val_df):,}")


print("\nLoading Random Forest model...")

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# =========================================================
# 1. CREATE SHAP SAMPLE
# =========================================================

fraud_df = val_df[
    val_df[TARGET] == 1
]

normal_df = val_df[
    val_df[TARGET] == 0
]


# Keep all fraud cases in the candidate pool,
# but use a manageable balanced sample for SHAP.

fraud_sample = fraud_df.sample(
    n=min(500, len(fraud_df)),
    random_state=RANDOM_STATE
)

normal_sample = normal_df.sample(
    n=500,
    random_state=RANDOM_STATE
)


shap_sample = pd.concat(
    [
        fraud_sample,
        normal_sample
    ],
    ignore_index=True
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"SHAP sample rows: {len(shap_sample):,}"
)

print(
    f"Fraud rows in sample: "
    f"{shap_sample[TARGET].sum():,}"
)


X_sample = shap_sample.drop(
    columns=[TARGET]
)


# =========================================================
# 2. APPLY SAVED PREPROCESSING
# =========================================================

print("\nApplying preprocessing...")

X_transformed = preprocessor.transform(
    X_sample
)

feature_names = (
    preprocessor.get_feature_names_out()
)


if hasattr(
    X_transformed,
    "toarray"
):
    X_transformed = X_transformed.toarray()


X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names
)


print(
    f"Transformed features: "
    f"{X_transformed_df.shape[1]}"
)


# =========================================================
# 3. SHAP EXPLAINER
# =========================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    model
)


print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    X_transformed_df,
    check_additivity=False
)


# Different SHAP versions may return:
# list[class0, class1]
# or ndarray with class dimension.

if isinstance(
    shap_values,
    list
):
    fraud_shap_values = shap_values[1]

elif (
    isinstance(shap_values, np.ndarray)
    and shap_values.ndim == 3
):
    fraud_shap_values = shap_values[:, :, 1]

else:
    fraud_shap_values = shap_values


print(
    "SHAP values calculated successfully."
)


# =========================================================
# 4. GLOBAL FEATURE IMPORTANCE
# =========================================================

mean_abs_shap = np.abs(
    fraud_shap_values
).mean(axis=0)


importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values(
    "mean_abs_shap",
    ascending=False
)


print("\n========================================")
print("TOP SHAP FEATURES")
print("========================================")

print(
    importance_df
    .head(15)
    .to_string(index=False)
)


importance_df.to_csv(
    MODEL_DIR / "shap_feature_importance.csv",
    index=False
)


# =========================================================
# 5. SHAP BAR PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    fraud_shap_values,
    X_transformed_df,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.title(
    "Random Forest - SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "shap_feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 6. SHAP SUMMARY PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    fraud_shap_values,
    X_transformed_df,
    max_display=15,
    show=False
)

plt.title(
    "Random Forest - SHAP Summary"
)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 7. LOCAL FRAUD EXPLANATION
# =========================================================

fraud_positions = np.where(
    shap_sample[TARGET].to_numpy() == 1
)[0]


if len(fraud_positions) > 0:

    fraud_position = fraud_positions[0]

    transaction = X_sample.iloc[
        fraud_position
    ]

    probability = pipeline.predict_proba(
        pd.DataFrame([transaction])
    )[0, 1]

    local_values = fraud_shap_values[
        fraud_position
    ]

    local_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": local_values,
        "feature_value": (
            X_transformed_df
            .iloc[fraud_position]
            .to_numpy()
        )
    })

    local_df["absolute_shap"] = np.abs(
        local_df["shap_value"]
    )

    local_df = local_df.sort_values(
        "absolute_shap",
        ascending=False
    )


    print("\n========================================")
    print("EXAMPLE FRAUD EXPLANATION")
    print("========================================")

    print(
        f"Fraud probability: "
        f"{probability:.6f}"
    )

    print("\nTop contributing features:")

    print(
        local_df[
            [
                "feature",
                "feature_value",
                "shap_value"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


    local_df.to_csv(
        MODEL_DIR
        / "example_fraud_shap_explanation.csv",
        index=False
    )


print("\nSaved:")
print(
    "models/shap_feature_importance.csv"
)
print(
    "models/example_fraud_shap_explanation.csv"
)

print("\nCharts saved:")
print(
    "images/shap_feature_importance.png"
)
print(
    "images/shap_summary.png"
)

print(
    "\nPhase 11 SHAP explainability completed successfully."
)
