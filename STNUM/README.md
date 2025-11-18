# 3D Digits Classification Project (PRML)

This project implements a full classification pipeline for recognizing
hand-written digits (0--9) drawn in the air and recorded as 3D motion
data. Each digit is stored as a CSV file containing an **N × 3**
sequence of\
(x, y, z) fingertip positions.

The goal is to implement the required function:

`C = digit_classify(testdata)`

where `testdata` is an **N × 3** motion and `C` is the predicted digit
(0--9).

This README explains **how the project works from start to finish** and\
**how to run every step.**

------------------------------------------------------------------------

# 📁 Project Structure

    STNUM/
    ├── code/
    │    ├── explore_data.py
    │    ├── extract_features.py
    │    ├── knn.py
    │    ├── train_knn.py
    │    ├── tune_knn.py
    │    ├── train_final_knn.py
    │    ├── digit_classify.py
    │    └── test_digit_classify.py
    └── data/
          ├── stroke_0_0001.csv
          ├── stroke_1_0034.csv
          ├── …

------------------------------------------------------------------------

# 🚀 **How the System Works (Step-by-Step)**

## **Step 1 --- Data: 3D Motions of Hand-Written Digits**

The folder `data/` contains 1000 CSV files.\
Each file contains an **N × 3** array of (x, y, z) coordinates recorded
while a person drew the digit in the air.\
N varies (e.g., 30--200 time steps).

------------------------------------------------------------------------

## **Step 2 --- Explore the Data**

Run:

    python explore_data.py

This script:

-   loads sample motions\
-   prints shapes and first rows\
-   plots:
    -   3D motion path\
    -   x(t), y(t), z(t) curves

------------------------------------------------------------------------

## **Step 3 --- Feature Extraction**

File: `extract_features.py`

Each raw motion (N × 3) is converted into a fixed-length numerical
feature vector:

1.  Normalize start point\
2.  Normalize scale\
3.  Resample to fixed length (e.g., 30 pts)\
4.  Flatten into 1D vector

------------------------------------------------------------------------

## **Step 4 --- Baseline kNN Classifier**

Run:

    python train_knn.py

This script:

-   loads all CSV files\
-   extracts features\
-   splits train/validation\
-   trains your custom kNN\
-   prints accuracy + confusion matrix

Baseline accuracy ≈ **95%**

------------------------------------------------------------------------

## **Step 5 --- Hyperparameter Tuning**

Run:

    python tune_knn.py

Searches:

-   `num_points` ∈ {30, 50, 70}\
-   `k` ∈ {1, 3, 5, 7, 9}

Best: **96%**, with `num_points = 30`, `k = 7`.

------------------------------------------------------------------------

## **Step 6 --- Train Final Model**

Run:

    python train_final_knn.py

Produces file:

    code/knn_model.npz

Contains:

-   X_train (1000 × 90)\
-   y_train\
-   num_points = 30\
-   k = 7

------------------------------------------------------------------------

## **Step 7 --- Final Digit Classifier**

The required function:

``` python
digit_classify(testdata) -> int
```

Steps:

1.  Load knn_model.npz\
2.  Extract features\
3.  Run kNN\
4.  Return predicted digit

Test:

    python test_digit_classify.py

------------------------------------------------------------------------

# 📌 Requirements

    pip install numpy pandas matplotlib

------------------------------------------------------------------------

# 🎉 Project Complete

This project now fully supports:

-   reading raw 3D motion data\
-   visualizing sample motions\
-   extracting normalized features\
-   training/tuning kNN\
-   saving final model\
-   predicting digits via `digit_classify()`

------------------------------------------------------------------------

If you'd like enhancements (badges, diagrams, accuracy section), just
ask!
