import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/paysim.csv")
PROCESSED_DIR = Path("data/processed")

TARGET = "isFraud"

DROP_COLUMNS = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud",
]


def load_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {df.shape[0]:,} rows")

    return df


def prepare_features(df):
    print("\nPreparing features...")

    X = df.drop(
        columns=DROP_COLUMNS + [TARGET]
    )

    y = df[TARGET]

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nTarget:")
    print(TARGET)

    return X, y


def split_data(X, y):
    print("\nSplitting dataset...")

    # First split:
    # 70% Train
    # 30% Temporary

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Second split:
    # Temporary 30% -> 15% Validation + 15% Test

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def print_split_info(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    print("\n=== DATASET SPLIT ===")

    print(f"Train:      {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Test:       {len(X_test):,}")

    print("\n=== FRAUD COUNTS ===")

    print(f"Train Fraud:      {y_train.sum():,}")
    print(f"Validation Fraud: {y_val.sum():,}")
    print(f"Test Fraud:       {y_test.sum():,}")

    print("\n=== FRAUD PERCENTAGES ===")

    print(
        f"Train:      {y_train.mean() * 100:.4f}%"
    )

    print(
        f"Validation: {y_val.mean() * 100:.4f}%"
    )

    print(
        f"Test:       {y_test.mean() * 100:.4f}%"
    )


def save_splits(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    print("\nSaving processed datasets...")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train = X_train.copy()
    train[TARGET] = y_train

    val = X_val.copy()
    val[TARGET] = y_val

    test = X_test.copy()
    test[TARGET] = y_test

    train.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    val.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )

    test.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print("Saved:")
    print("data/processed/train.csv")
    print("data/processed/validation.csv")
    print("data/processed/test.csv")


def main():

    df = load_data()

    X, y = prepare_features(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    print_split_info(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    save_splits(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
