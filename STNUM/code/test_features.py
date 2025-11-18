import os
from explore_data import load_motion as load_motion   # you can rename this later
from extract_features import extract_features

# Load one example motion file
path = os.path.join("..", "data", "stroke_0_0001.csv")
motion = load_motion(path)

features = extract_features(motion, num_points=50)

print("Original motion shape:", motion.shape)
print("Feature vector shape:", features.shape)
print("First 10 feature values:", features[:10])