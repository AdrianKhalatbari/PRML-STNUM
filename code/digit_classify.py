import os
import numpy as np

from extract_features import extract_features
from knn import knn

"""
This module implements the required `digit_classify(testdata)` function for the
3D digit recognition project. The classifier uses a k-Nearest Neighbours (kNN)
model trained offline and stored in `knn_model.npz`. The stored model contains:

    - X_train: feature matrix for all training motions
    - y_train: corresponding digit labels
    - num_points: number of resampled points used in feature extraction
    - k: number of neighbours used by the kNN classifier

The function `digit_classify()` loads these parameters, extracts features from a
new input motion, and returns the predicted digit (0_9). The function must not
print, plot, or perform training, as required by the project specification.
"""

def _load_model():
    """
    Load kNN 'model' from knn_model.npz.
    This is just the stored training data and hyperparameters.
    """

    model_path = os.path.join(os.path.dirname(__file__), "knn_model.npz")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Run train_final_knn.py first to create it."
        )

    data = np.load(model_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    num_points = int(data["num_points"])
    k = int(data["k"])

    _MODEL_CACHE = {
        "X_train": X_train,
        "y_train": y_train,
        "num_points": num_points,
        "k": k,
    }
    return _MODEL_CACHE


def digit_classify(testdata):
    """
    Classify a single digit motion using kNN.

    Parameters
    ----------
    testdata : array-like, shape (N, 3)
        Motion of one digit: N time steps, each with (x, y, z).

    Returns
    -------
    int
        Predicted digit in {0, 1, ..., 9}.
    """
    # Ensure input is a NumPy array
    motion = np.asarray(testdata, dtype=float)

    # Load model (training data + settings)
    model = _load_model()
    X_train = model["X_train"]
    y_train = model["y_train"]
    num_points = model["num_points"]
    k = model["k"]

    # Extract features for this motion
    features = extract_features(motion, num_points=num_points)  # shape (D,)
    features = features.reshape(1, -1)  # shape (1, D) for kNN

    # kNN prediction
    y_pred = knn(X_train, y_train, features, k)

    # Return as plain Python int
    return int(y_pred[0])