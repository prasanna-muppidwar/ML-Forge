import numpy as np

def mean_squared_error(y_true, y_pred):
    # Convert input to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the squared differences
    squared_diff = (y_true - y_pred) ** 2
    
    # Return the mean of these squared differences
    return np.mean(squared_diff)
