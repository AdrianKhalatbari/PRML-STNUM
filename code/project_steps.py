from explore_data import explore_data
from tune_knn import tune_knn
from train_final_knn import train_final_knn
from test_digit_classify import test_digit_classify
from test_features import test_feature
from train_knn import train_knn

def main():
    # ------------ Step 1 — Explore the Data ------------ #
    """
    Explore the motion-capture dataset.

    This step loads basic information about the available motion files and, if desired,
    visualizes sample motions to better understand their structure and characteristics.
    To inspect details or generate plots, set the DEBUG flag to True in
    `explore_data.py`.

    scripts:
        - explore_data.py
    """
    explore_data()


    # ------------ Step 2 — Feature Extraction ------------ #
    """
    Convert each raw motion sequence into a fixed-length numerical
    feature vector. This applies normalization, scale adjustment, resampling
    to a fixed number of points, and flattening into a 1D vector suitable for
    machine learning models.
    This function does not pprint anything by itself; it returns the feature vectors
    for further use.
    To test the feature extraction on a sample motion array, set the DEBUG flag to True
    in `test_features.py` and call the function as shown below:

    Scripts:
        - extract_features.py
        - test_features.py
    """
    test_feature()

    # ------------ Step 3 — kNN Classifier, and Hyperparameter Tuning ------------ #
    """
    Tune the k-Nearest Neighbors (kNN) classifier over:
    kNN contains 3 distinct parts:
        1. Implement knn functionality
        2. Train/validation split
        3. Hyperparameter tuning over the number of points per motion

    tune_knn does not print anything, to view the tuning results, uncomment the print statements in tune_knn.py.    

    Scripts:
        - knn.py
        - tune_knn.py
        - train_knn.py
    """
    train_knn()
    tune_knn()

    # ------------ Step 4 - Train Final Model and Evaluate on Test Set ------------ #
    """
    Train the final kNN model using the best hyperparameters found during tuning,
    and evaluate its performance on the held-out test set.
    This step reports the final test accuracy of the model.

    Scripts:
        - train_final_knn.py
    """
    train_final_knn()


    # ------------ Step 5 - Final Digit Classifier ------------ #
    """
    Using the trained kNN model, classify new digit motion sequences and report accuracy.
    This step loads the saved kNN model and applies it to the test digit motions.
    The main function `digit_classify()` handles loading the model, processing the test data,
    and outputting the predicted digits along with accuracy metrics. This function does not
    print anything by itself; it returns the predictions for further use.
    To run a simple test of the digit classifier, call the function below.

    File: digit_classify.py
        This file contains the main function `digit_classify()` which:
        - Loads the trained kNN model
        - Loads and processes the test digit motion data
        - Uses the kNN model to predict the digit labels
        

    Scripts:
        - digit_classify.py
            This file contains the main function `digit_classify()` which:
            - Loads the trained kNN model
            - Loads and processes the test digit motion data
            - Uses the kNN model to predict the digit labels
        - test_digit_classify.py
            This file tests the `digit_classify()` function on a sample motion file.
            It prints the predicted digit for verification.
    """
    test_digit_classify()


if __name__ == "__main__":
    main()
