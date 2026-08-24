import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["orig_balance_change"] = (
            X["oldbalanceOrg"]
            - X["newbalanceOrig"]
        )

        X["dest_balance_change"] = (
            X["newbalanceDest"]
            - X["oldbalanceDest"]
        )

        X["orig_balance_error"] = (
            X["oldbalanceOrg"]
            - X["amount"]
            - X["newbalanceOrig"]
        )

        X["dest_balance_error"] = (
            X["oldbalanceDest"]
            + X["amount"]
            - X["newbalanceDest"]
        )

        X["amount_to_orig_balance"] = np.where(
            X["oldbalanceOrg"] > 0,
            X["amount"] / X["oldbalanceOrg"],
            0.0
        )

        X["orig_account_emptied"] = (
            (X["oldbalanceOrg"] > 0)
            & (X["newbalanceOrig"] == 0)
        ).astype(int)

        return X
