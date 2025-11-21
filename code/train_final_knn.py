import os
import numpy as np
from train_knn import build_dataset

"""
    Builds the final k-Nearest Neighbors (kNN) dataset and saves the trained
    model data into a single `.npz` file.
    
    This script constructs the full training dataset using `build_dataset()`,
    prints its shape for verification, and then stores all components needed
    for later classification. The saved file, `knn_model.npz`, acts as a
    compact "package" containing multiple NumPy arrays and parameters.
    
    About `.npz` files:
        • `.npz` stores multiple arrays (and scalars) together, similar to a
          tiny ZIP archive. (Also there is another type named `.npy`, it stores a single NumPy array.)
    
    The file `knn_model.npz` created by this script contains:
        • X_train: the full training feature matrix
        • y_train: the training labels
        • num_points: number of points used to build the dataset
        • k: the selected value of K for kNN
    
    Later, other scripts (e.g., `digit_classify.py`) can load this data using:
    
        data = np.load("knn_model.npz")
        X_train = data["X_train"]
        y_train = data["y_train"]
        num_points = int(data["num_points"])
        k = int(data["k"])
    
    This allows the classification script to reconstruct the trained kNN
    model without retraining, simply by loading this packaged dataset.
"""


# Final hyperparameters from tuning
# This values should match the best parameters found during tuning and come from tune_knn.py
# To view the best parameters, uncomment the print statements in tune_knn.py
FINAL_NUM_POINTS = 30
FINAL_K = 7
DEBUG = False


def train_final_knn():
    # Build final dataset with selected num_points
    X, y = build_dataset(num_points=FINAL_NUM_POINTS)

    # Print dataset shape
    # To view dataset shape, set DEBUG = True
    if DEBUG:
        print("Final training set shape:")
        print("  X:", X.shape)
        print("  y:", y.shape)

    # Save model parameters (training data + settings)
    model = {
        "X_train": X,
        "y_train": y,
        "num_points": FINAL_NUM_POINTS,
        "k": FINAL_K,
    }

    # Save final kNN model to .npz file
    model_path = os.path.join(os.path.dirname(__file__), "knn_model.npz")
    np.savez(model_path, **model)


if __name__ == "__main__":
    train_final_knn()