import numpy as np

def mse(yhat: np.ndarray, y: np.ndarray):
    m = len(y)
    cost = np.sum(np.square(yhat - y)) / (2 * m)
    return cost