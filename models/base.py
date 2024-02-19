import numpy as np


class MLForecastModel:

    def __init__(self) -> None:
        self.fitted = False

    def fit(self, train_X: np.ndarray, val_X=None, flag=None) -> None:
        """
        :param X: history timesteps
        :param Y: future timesteps to predict
        """
        self._fit(train_X, val_X, flag)
        self.fitted = True

    def _fit(self, X: np.ndarray, val_X=None, flag=None):
        raise NotImplementedError

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: history timesteps
        :return: predicted future timesteps
        """
        if not self.fitted:
            raise ValueError("Model has not been trained.")
        pred = self._forecast(X)
        return pred