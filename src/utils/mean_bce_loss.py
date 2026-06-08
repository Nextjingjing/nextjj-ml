import numpy as np

def mean_bce_loss(yhat: np.ndarray, y: np.ndarray) -> float:
    m = len(yhat)
    yhat = np.clip(yhat, 1e-15, 1 - 1e-15) # Protect log(0)
    return - np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / m