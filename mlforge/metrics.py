import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error to evaluate the model performance.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    squared_diff = (y_true - y_pred) ** 2
    return np.mean(squared_diff)

def r2score(y_true, y_pred):
    """
    Calculate the R-squared score to evaluate the model performance.
    """
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2