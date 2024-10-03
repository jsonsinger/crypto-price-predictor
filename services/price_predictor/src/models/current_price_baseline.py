import pandas as pd


class CurrentPriceBaseline:
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model to the training data
        """
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the current price as the next price
        """
        return X["close"]
