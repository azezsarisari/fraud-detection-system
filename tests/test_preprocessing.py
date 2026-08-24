import pandas as pd

from src.transformers import FraudFeatureEngineer


def test_feature_engineering_columns():

    df = pd.DataFrame([
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
    ])

    transformer = FraudFeatureEngineer()

    result = transformer.transform(df)

    expected_columns = [
        "orig_balance_change",
        "dest_balance_change",
        "orig_balance_error",
        "dest_balance_error",
        "amount_to_orig_balance",
        "orig_account_emptied"
    ]

    for column in expected_columns:
        assert column in result.columns


def test_fraud_feature_values():

    df = pd.DataFrame([
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
    ])

    transformer = FraudFeatureEngineer()

    result = transformer.transform(df)

    row = result.iloc[0]

    assert row["orig_balance_change"] == 181.0

    assert row["orig_balance_error"] == 0.0

    assert row["dest_balance_error"] == 181.0

    assert row["amount_to_orig_balance"] == 1.0

    assert row["orig_account_emptied"] == 1


def test_zero_sender_balance():

    df = pd.DataFrame([
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 100.0,
            "oldbalanceOrg": 0.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
    ])

    transformer = FraudFeatureEngineer()

    result = transformer.transform(df)

    assert (
        result.iloc[0]["amount_to_orig_balance"]
        == 0.0
    )

    assert (
        result.iloc[0]["orig_account_emptied"]
        == 0
    )
