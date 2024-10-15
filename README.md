

![Repository Visitors](https://komarev.com/ghpvc/?username=prasanna-muppidwar&color=blue&label=Repository+Visitors)

**ML Forge** is an easy-to-use, open-source machine learning library designed to help you build machine learning models from scratch. It currently supports linear regression and evaluation metrics, with an intuitive interface for training and predicting models.

## Features
- Supervised Machine Learning algorithms implemented with ease of Understanding.
- Evaluation Metrics added!
- Simple and flexible to understand and implement.

## Installation

Clone the repository directly from GitHub:

```bash
git clone "https://github.com/prasanna-muppidwar/mlforge.git"
```

## Documentation

### Implement Linear Regression 
<details>
<summary>Linear Regression </summary>
Linear Regression is a simple machine learning algorithm used to model the relationship between a dependent variable and one or more independent variables. It tries to find the best-fitting straight line (called the regression line) by minimizing the error between predicted and actual values.
</details>

``` python
import numpy as np
import pandas as pd
from mlforge.train_test_split import train_test_split
from mlforge.preprocessing import StandardScaler
from mlforge.linear_regression import LinearRegression
from mlforge.metrics import r2score, mean_squared_error

df = pd.read_csv("hf://datasets/scikit-learn/auto-mpg/auto-mpg.csv")

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()

X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']].values
y = df['mpg'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

accuracy = r2score(y_test, predictions)
print("Accuracy (R² score):", accuracy)

mse_value = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse_value)

```


