
import numpy as np
import pytest
from src.models.linear_regression import LinearRegression

class TestLinearRegression:
    @pytest.fixture
    def setup_data(self):
        model = LinearRegression()
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = 2 * X + 10
        X = X.reshape(-1, 1)
        return model, X, y

    def test_compute_gradient(self, setup_data):
        model, X, y = setup_data
        
        model.w = np.array([5.0])
        model.b = 5
        yhat = model.predict(X)
        dJ_dw, dJ_db = model._compute_gradient(X, yhat, y)

        assert dJ_dw[0] == pytest.approx(88.0)
        assert dJ_db == pytest.approx(11.5)
    
    def test_fit(self, setup_data):
        model, X, y = setup_data
        model.fit(X, y, epoch=2000, _lamda=0.01)

        assert model.w[0] == pytest.approx(2.0, abs=0.5)
        assert model.b == pytest.approx(10.0, abs=0.5)

    def test_predict(self, setup_data):
        model, X, y = setup_data
        model.fit(X, y, epoch=2000, _lamda=0.01)
        y_pred = model.predict(np.array([1, 2, 3]))

        assert y_pred[0] == pytest.approx(12, abs=0.2)
        assert y_pred[1] == pytest.approx(14, abs=0.2)
        assert y_pred[2] == pytest.approx(16, abs=0.2)
