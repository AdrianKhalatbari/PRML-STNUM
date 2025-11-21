import numpy as np
from knn import knn
from train_knn import build_dataset, train_val_split

DEBUG = False

"""
    We performed a grid search over the number of resampled points
    (num_points ∈ {30, 50, 70})
    and the number of neighbours in kNN (k ∈ {1, 3, 5, 7, 9}).
    
    The best validation accuracy (96.0%) was achieved for several combinations,
    all including k=7.
    
    The configuration (num_points=30, k=7) achieved high accuracy while producing
    the smallest feature dimension (90 features), so we selected this setting
    for our final model.

    Steps:
    1. Build dataset with given num_points
    2. Split into train/validation sets
    3. Run kNN with given k
    4. Compute and return validation accuracy
"""
def evaluate_knn_for_config(num_points, k, val_ratio=0.2, seed=42):
    """
    Build dataset with given num_points, split into train/val,
    run kNN with given k, and return validation accuracy.
    """
    # Build dataset (features + labels)
    X, y = build_dataset(num_points=num_points)

    # Train/validation split
    X_train, y_train, X_val, y_val = train_val_split(
        X, y, val_ratio=val_ratio, seed=seed
    )

    # Run kNN
    y_pred = knn(X_train, y_train, X_val, k)

    # Compute accuracy
    accuracy = np.mean(y_pred == y_val)
    return accuracy


def tune_knn():
    # Hyperparameter grids
    k_values = [1, 3, 5, 7, 9]
    num_points_values = [30, 50, 70]

    results = []

    for num_points in num_points_values:
        for k in k_values:
            acc = evaluate_knn_for_config(num_points=num_points, k=k)
            results.append((num_points, k, acc))
            # To view all results, set DEBUG = True
            if DEBUG:
                print(f"num_points={num_points:3d}, k={k:2d} -> accuracy={acc:.4f}")

    # Find best configuration
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    best_num_points, best_k, best_acc = results_sorted[0]

    # Print summary of results
    # To view all results, set DEBUG = True
    if DEBUG:
        print("\nBest configuration:")
        print(f"  num_points = {best_num_points}")
        print(f"  k          = {best_k}")
        print(f"  accuracy   = {best_acc:.4f}")

        print("\nAll results (sorted by accuracy):")
        for num_points, k, acc in results_sorted:
            print(f"  num_points={num_points:3d}, k={k:2d} -> accuracy={acc:.4f}")


if __name__ == "__main__":
    tune_knn()