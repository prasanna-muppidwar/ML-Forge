#Add preprocessing techniques here

import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Calculate the mean and standard deviation for each feature in X."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        """Scale the dataset using the calculated mean and standard deviation."""
        if self.mean is None or self.std is None:
            raise Exception("You must fit the scaler before transforming the data.")
        
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)
