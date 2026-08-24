import pandas as pd
import numpy as np
from pathlib import Path


PROCESSED_DIR = Path("data/processed")


def add_features(df):
    df = df.copy()

    # Sender balance change
    df["orig_balance_change"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    # Receiver balance change
    df["dest_balance_change"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    # Expected sender balance:
    # old balance - transaction amount = new balance
    df["orig_balance_error"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    # Expected receiver balance:
    # old balance + transaction amount = new balance
    df["dest_balance_error"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    # Transaction amount relative to sender's balance
    df["amount_to_orig_balance"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        0.0
    )

    # Whether transaction emptied sender account
    df["orig_account_emptied"] = (
        (df["oldbalanceOrg"] > 0)
        & (df["newbalanceOrig"] == 0)
    ).astype(int)

    return df


def process_file(filename):
    input_path = PROCESSED_DIR / filename

    print(f"\nLoading {input_path}...")

    df = pd.read_csv(input_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns before: {len(df.columns)}")

    df = add_features(df)

    print(f"Columns after: {len(df.columns)}")

    df.to_csv(
        input_path,
        index=False
    )

    print(f"Updated: {input_path}")


def main():

    print("Starting feature engineering...")

    process_file("train.csv")
    process_file("validation.csv")
    process_file("test.csv")

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()
