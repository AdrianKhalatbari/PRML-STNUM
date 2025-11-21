import os
from explore_data import load_motion
from digit_classify import digit_classify

def test_digit_classify():
    # pick any example file
    path = os.path.join("..", "data", "stroke_0_0001.csv")
    motion = load_motion(path)

    pred = digit_classify(motion)
    print("Predicted digit:", pred)

if __name__ == "__main__":
    test_digit_classify()