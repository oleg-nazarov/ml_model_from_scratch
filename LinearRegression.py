import numpy as np
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self):
        self.fitted = False
        self.w = None
        self.w0 = None

    # supposed X and y to be np.array
    def fit(self, X, y):
        # W = (X(T) * X)(-1) * X(T) * y
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        weights = np.dot(np.linalg.inv(X.T @ X) @ X.T, y)

        self.w = weights[:-1]
        self.w0 = weights[-1]

        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise Exception('Call "fit" before "predict"')

        return np.dot(X, self.w) + self.w0

    def score(self, X, y):
        pred = self.predict(X)

        return r2_score(y, pred)
