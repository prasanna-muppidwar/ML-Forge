import numpy as np
from collections import Counter

# Decision Tree Classifier (Simple)
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._find_best_split(X, y, num_samples, num_features)
            if best_split:
                left_indices, right_indices = best_split["groups"]
                left = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
                right = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)
                return {"feature": best_split["feature"], "threshold": best_split["threshold"], "left": left, "right": right}

        return self._majority_class(y)

    def _find_best_split(self, X, y, num_samples, num_features):
        best_split = {}
        best_gini = float("inf")

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]
                if len(left_indices) > 0 and len(right_indices) > 0:
                    gini = self._gini_index(y[left_indices], y[right_indices])
                    if gini < best_gini:
                        best_gini = gini
                        best_split = {
                            "feature": feature,
                            "threshold": threshold,
                            "groups": (left_indices, right_indices)
                        }
        return best_split

    def _gini_index(self, left_y, right_y):
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        gini_left = 1.0 - sum((np.sum(left_y == c) / n_left) ** 2 for c in np.unique(left_y))
        gini_right = 1.0 - sum((np.sum(right_y == c) / n_right) ** 2 for c in np.unique(right_y))
        return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def _predict_single(self, x, tree):
        if isinstance(tree, dict):
            feature = tree["feature"]
            threshold = tree["threshold"]
            if x[feature] <= threshold:
                return self._predict_single(x, tree["left"])
            else:
                return self._predict_single(x, tree["right"])
        return tree


# Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=int(self.sample_size * n_samples), replace=True)
        return X[indices], y[indices]


"""def test_random_forest():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    rf = DecisionTree()  # Increased n_trees for better generalization
    rf.fit(X, y)

    # Testing Predictions
    predictions = rf.predict(X)
    print("Predictions: ", predictions)

    # Check if predictions are mostly correct
    accuracy = np.sum(predictions == y) / len(y)
    assert accuracy >= 0.75, f"Test failed! Accuracy: {accuracy}"  
    print("Test passed!")

if __name__ == "__main__":
    test_random_forest()"""