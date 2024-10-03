import pandas as pd


class MovingAverageBaseline:
    """
    A simple baseline model that predicts the moving average of the last `window_size` prices
    """

    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model to the training data
        """
        pass

    def predict(self, X):
        pass
