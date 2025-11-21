import numpy as np

"""
    This file contains functions to extract features from a motion file.
        1.	Normalize position
        2.	Scale to remove size differences
        3.	Resample to fixed number of points
        4.	Flatten into one feature vector
"""

def resample_motion(motion, num_points=50):
    """
    Linearly resample a motion (N x 3 array) to a fixed number of points.

    motion: N x 3 numpy array
    num_points: how many points the output should have
    """
    N = motion.shape[0]
    if N == num_points:
        return motion

    # Original time indices scaled to [0, 1]
    t_original = np.linspace(0, 1, N)
    t_resampled = np.linspace(0, 1, num_points)

    motion_resampled = np.zeros((num_points, 3))

    # Interpolate each coordinate separately
    for dim in range(3):
        motion_resampled[:, dim] = np.interp(
            t_resampled, t_original, motion[:, dim]
        )

    return motion_resampled


def extract_features(motion, num_points=50):
    """
    Convert variable-length motion (N x 3) into a fixed-length feature vector.

    Steps:
    1. Normalize starting position
    2. Normalize scale
    3. Resample to fixed number of points
    4. Flatten into a 1D vector
    """

    motion = motion.astype(float)

    # 1. Normalize starting position (shift so motion begins at 0,0,0)
    motion = motion - motion[0]

    # 2. Normalize scale (make all motions comparable in size)
    max_range = np.max(np.linalg.norm(motion, axis=1))
    if max_range > 0:
        motion = motion / max_range

    # 3. Resample to fixed length
    motion_fixed = resample_motion(motion, num_points)

    # 4. Flatten to a 1D vector (length = num_points * 3)
    features = motion_fixed.flatten()

    return features