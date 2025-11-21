import os
from explore_data import load_motion as load_motion
from extract_features import extract_features

DEBUG = False

def test_feature():
    """
    Test the feature extraction on a sample motion array.
    This function loads a sample motion file, extracts features using
    the `extract_features` function, and prints the shape of the resulting
    feature vector along with some sample values.

    Steps:
    1. Load a sample motion file. You can change the path to test different files.
    2. Extract features using `extract_features`.
    3. Print the shape and sample values of the feature vector.
    """
    # Load one example motion file
    path = os.path.join("..", "data", "stroke_0_0001.csv")
    motion = load_motion(path)

    features = extract_features(motion, num_points=50)

    # To view the results, set DEBUG to True at the top of this file
    if DEBUG:
        print("Original motion shape:", motion.shape)
        print("Feature vector shape:", features.shape)
        print("First 10 feature values:", features[:10])

if __name__ == "__main__":
    test_feature()