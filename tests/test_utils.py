import numpy as np
import pytest
from src.utils.mse import mse

@pytest.mark.parametrize("yhat, y , expected", [
    (np.array([1, 2]), np.array([1, 2]), 0.0),
    (np.array([2, 3, 4, 5]), np.array([1, 1, 1, 1]), 3.75)
])
def test_mse(yhat, y, expected):
    assert mse(yhat, y) == pytest.approx(expected)