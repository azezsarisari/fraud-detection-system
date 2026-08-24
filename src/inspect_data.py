import pandas as pd

FILE_PATH = "data/raw/paysim.csv"

print("Loading dataset...")
df = pd.read_csv(FILE_PATH)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== FRAUD DISTRIBUTION ===")
print(df["isFraud"].value_counts())

print("\n=== FRAUD PERCENTAGE ===")
fraud_percentage = df["isFraud"].mean() * 100
print(f"{fraud_percentage:.4f}%")

print("\n=== TRANSACTION TYPES ===")
print(df["type"].value_counts())
