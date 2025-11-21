import numpy as np

def euclidean_distance(a, b):
    """Compute Euclidean distance between two 1D vectors."""
    return np.sqrt(np.sum((a - b) ** 2))


def knn(train_data, train_labels, test_data, k):
    """
    k-Nearest Neighbours classifier (multi-sample).

    Parameters
    ----------
    train_data : array, shape (n_train, d)
        Feature vectors for training samples.
    train_labels : array, shape (n_train,)
        Class labels for training samples.
    test_data : array, shape (n_test, d)
        Feature vectors for test samples.
    k : int
        Number of neighbours.

    Returns
    -------
    predictions : array, shape (n_test,)
        Predicted labels for each test sample.
    """
    predictions = []

    for test_point in test_data:
        # Compute distances to all training points
        # (vectorized version is faster, but this is clear and simple)
        distances = [euclidean_distance(test_point, x) for x in train_data]

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Get labels of k nearest neighbors
        k_labels = train_labels[k_indices]

        # Majority vote
        values, counts = np.unique(k_labels, return_counts=True)
        predictions.append(values[np.argmax(counts)])

    return np.array(predictions)