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

## Docs 

### Implement Linear Regression 

``` bash 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from algokit.linear_regression import LinearRegression, r2_score



X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)



accu = r2_score(y_test, predictions)
print("Accuracy:", accu)


```
### train test split
```bash 
import numpy as np
from algokit.linear_regression import LinearRegression, r2_score
from algokit.train_test_split import train_test_split
from sklearn import datasets

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

accu = r2_score(y_test, predictions)
print("Accuracy (R² score):", accu)

```