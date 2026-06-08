import numpy as np
import pytest
from src.utils.mse import mse
from src.utils.sigmoid import sigmoid
from src.utils.mean_bce_loss import mean_bce_loss

@pytest.mark.parametrize("yhat, y, expected", [
    (np.array([1, 2]), np.array([1, 2]), 0.0),
    (np.array([2, 3, 4, 5]), np.array([1, 1, 1, 1]), 3.75)
])
def test_mse(yhat, y, expected):
    assert mse(yhat, y) == pytest.approx(expected)

@pytest.mark.parametrize("z, expected", [
    (2, 0.8807970779779563),
    (-3, 0.04742587317751919),
    (-3.2, 0.039165722796722056),
    (3.2, 0.960834277203278)
])
def test_sigmoid(z, expected):
    assert sigmoid(z) == pytest.approx(expected)

@pytest.mark.parametrize("yhat, y, expected", [
    (np.array([0.1, 0.85, 0.05]), np.array([1, 0, 0]), 1.416999453),
    (np.array([0.15, 0.95, 0.12]), np.array([0, 1, 0]), 0.113881458),
])
def test_mean_bce_loss(yhat, y, expected):
    assert mean_bce_loss(yhat, y) == pytest.approx(expected, abs=1e-5)