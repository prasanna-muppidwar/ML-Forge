import numpy as np

def mean_squared_error(y_true, y_pred):
    # Convert input to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the squared differences
    squared_diff = (y_true - y_pred) ** 2
    
    # Return the mean of these squared differences
    return np.mean(squared_diff)

def r2_score(y_true, y_pred):
    """
    Calculate the R-squared score to evaluate the model performance.
    """
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2