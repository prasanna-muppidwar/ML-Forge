**AlgoKit** is an easy-to-use, open-source machine learning library designed to help you build machine learning models from scratch. It currently supports linear regression and evaluation metrics, with an intuitive interface for training and predicting models.

## Features
- Linear Regression implemented with gradient descent.
- R-squared score calculation for evaluating models.
- Simple and flexible API for integrating into any project.

## Installation

To install directly from the GitHub repository:

```bash
pip install git+https://github.com/prasanna-muppidwar/AlgoKit.git
```

## Usage
```
from algokit import LinearRegression, r2_score
import numpy as np

# Example Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])

# Initialize and train the model
model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Calculate R-squared
score = r2_score(y, predictions)
print("R2 Score:", score)

```
