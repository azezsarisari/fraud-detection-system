import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "data/raw/paysim.csv"
IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

print("Loading PaySim dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape[0]:,} rows")


# =========================================================
# 1. FRAUD BY TRANSACTION TYPE
# =========================================================

print("\n=== FRAUD BY TRANSACTION TYPE ===")

fraud_by_type = pd.crosstab(
    df["type"],
    df["isFraud"],
    margins=True
)

print(fraud_by_type)


# =========================================================
# 2. FRAUD RATE BY TRANSACTION TYPE
# =========================================================

print("\n=== FRAUD RATE BY TRANSACTION TYPE ===")

fraud_rate = (
    df.groupby("type")["isFraud"]
    .agg(["count", "sum", "mean"])
    .rename(
        columns={
            "count": "total_transactions",
            "sum": "fraud_transactions",
            "mean": "fraud_rate"
        }
    )
)

fraud_rate["fraud_percentage"] = (
    fraud_rate["fraud_rate"] * 100
)

fraud_rate = fraud_rate.sort_values(
    "fraud_percentage",
    ascending=False
)

print(
    fraud_rate[
        [
            "total_transactions",
            "fraud_transactions",
            "fraud_percentage"
        ]
    ]
)


# =========================================================
# 3. AMOUNT ANALYSIS
# =========================================================

print("\n=== NORMAL TRANSACTION AMOUNTS ===")

normal_amount = df.loc[
    df["isFraud"] == 0,
    "amount"
]

print(normal_amount.describe())


print("\n=== FRAUD TRANSACTION AMOUNTS ===")

fraud_amount = df.loc[
    df["isFraud"] == 1,
    "amount"
]

print(fraud_amount.describe())


print("\n=== MEDIAN AMOUNT ===")

print(
    df.groupby("isFraud")["amount"].median()
)


# =========================================================
# 4. FRAUD BALANCE ANALYSIS
# =========================================================

print("\n=== FRAUD BALANCE ANALYSIS ===")

fraud_df = df[df["isFraud"] == 1]

balance_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

print(
    fraud_df[balance_columns].describe()
)


# =========================================================
# 5. CHECK WHICH TYPES CONTAIN FRAUD
# =========================================================

print("\n=== FRAUD TRANSACTION TYPES ===")

print(
    fraud_df["type"].value_counts()
)


# =========================================================
# 6. VISUALIZATION — TRANSACTION TYPES
# =========================================================

transaction_counts = df["type"].value_counts()

plt.figure(figsize=(9, 5))

transaction_counts.plot(
    kind="bar"
)

plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "transaction_type_distribution.png",
    dpi=150
)

plt.close()


# =========================================================
# 7. VISUALIZATION — FRAUD BY TYPE
# =========================================================

fraud_type_counts = fraud_df["type"].value_counts()

plt.figure(figsize=(8, 5))

fraud_type_counts.plot(
    kind="bar"
)

plt.title("Fraud Transactions by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Fraud Transactions")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "fraud_by_transaction_type.png",
    dpi=150
)

plt.close()


# =========================================================
# 8. VISUALIZATION — FRAUD RATE
# =========================================================

plt.figure(figsize=(9, 5))

fraud_rate["fraud_percentage"].plot(
    kind="bar"
)

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Percentage (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "fraud_rate_by_type.png",
    dpi=150
)

plt.close()


print("\nEDA completed successfully.")

print("\nCharts saved to:")
print("images/transaction_type_distribution.png")
print("images/fraud_by_transaction_type.png")
print("images/fraud_rate_by_type.png")

