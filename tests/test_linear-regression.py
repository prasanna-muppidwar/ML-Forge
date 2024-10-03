import unittest
import numpy as np
from algokit.linear_regression import LinearRegression, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Generate synthetic regression data for testing
        self.X, self.y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        
        # Initialize the model
        self.model = LinearRegression(learning_rate=0.01, n_iters=1000)

    def test_fit(self):
        # Ensure that fitting the model doesn't raise errors
        self.model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)

    def test_predict(self):
        # Test that the model can make predictions after fitting
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertIsInstance(predictions, np.ndarray)

    def test_r2_score(self):
        # Test the R-squared function for accuracy
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, predictions)
        self.assertGreaterEqual(r2, 0)
        self.assertLessEqual(r2, 1)

    def test_mean_squared_error(self):
        # Test the model's mean squared error function
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        mse = np.mean((self.y_test - predictions) ** 2)
        self.assertGreaterEqual(mse, 0)

if __name__ == "__main__":
    unittest.main()
