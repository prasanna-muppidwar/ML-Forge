import unittest
import numpy as np
from algokit.linear_regression import LinearRegression, r2_score

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.model = LinearRegression(learning_rate=0.01, n_iters=1000)

    def test_fit_predict(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([5, 7, 9, 11, 13])
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertAlmostEqual(np.mean((y - predictions) ** 2), 0, delta=1)

    def test_r2_score(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        self.assertEqual(r2_score(y_true, y_pred), 1.0)

    def test_empty_input(self):
        X = np.array([]).reshape(0, 1)
        y = np.array([])
        with self.assertRaises(ValueError):
            self.model.fit(X, y)

if __name__ == '__main__':
    unittest.main()
