import numpy as np
from ..utils.sigmoid import sigmoid
from ..utils.mean_bce_loss import mean_bce_loss

class LogisticRegression:
    def __init__(self):
        self.w: np.ndarray = None
        self.b: float = None

    def fit(self, X: np.ndarray, y: np.ndarray, epoch: int = 1000, _lamda: float = 0.001) -> None:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0.
        for _ in range(epoch):
            yhat = self.predict(X)
            dJ_dw, dJ_db = self._compute_gradient(X, yhat, y)
            self.w = self.w - (_lamda * dJ_dw)
            self.b = self.b - (_lamda * dJ_db)

            if (_ + 1) % 500 == 0:
                print(f"Epoch {_ + 1} | Cost function: {mean_bce_loss(yhat ,y)}")
                print("w:", self.w)
                print("b:", self.b)


    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def _compute_gradient(self, X: np.ndarray, yhat: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        m, n = X.shape
        err = yhat - y
        dJ_db = np.sum(err) / m
        dJ_dw = np.zeros(n)
        for j in range(n):
            dJ_dw[j] = np.sum(err * X[:, j]) / m
        return dJ_dw , dJ_db