import numpy as np

def r2_score(y_true, y_pred):
    """
    Calculate the R-squared score to evaluate the model performance.
    """
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Linear Regression model using gradient descent.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            The step size for gradient descent optimization.
        n_iters : int, default=1000
            Number of iterations to optimize weights.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the linear regression model using input features and target values.
        
        Parameters:
        -----------
        X : np.ndarray
            Input feature matrix (n_samples, n_features).
        y : np.ndarray
            Target values (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict target values using the trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Input feature matrix (n_samples, n_features).
            
        Returns:
        --------
        y_pred : np.ndarray
            Predicted target values (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias
