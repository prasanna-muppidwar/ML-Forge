import unittest
import numpy as np
from ml-forge.logistic_regression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression(learning_rate=0.01, n_iters=1000)

    def test_fit_predict(self):
        X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        y = np.array([0, 1, 1, 0])
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertAlmostEqual(np.mean((y - predictions) ** 2), 0, delta=1)

    def test_binary_classification(self):
        X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
        y = np.array([0, 0, 1, 1])
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertTrue(np.all(predictions == np.array([0, 0, 1, 1])))

    def test_empty_input(self):
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        with self.assertRaises(ValueError):
            self.model.fit(X, y)

if __name__ == '__main__':
    unittest.main()
