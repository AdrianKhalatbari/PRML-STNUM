import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
This file contains functions to:
    — Load the data
    — Print shapes
    — Visualize a few motions
    — Help you understand the data for the report
    Note: visualizations are optional. I put them behind a DEBUG flag.
    You can enable them by setting DEBUG = True below.
"""

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEBUG = False


def load_motion(csv_path):
    """
    Load a single motion from CSV.

    Assumes: each row = one time step, columns = x, y, z.
    If there are more than 3 columns, we only keep the first 3.

    Returns:
        motion: NumPy array of shape (N, 3)
    """
    # Read with pandas, then convert to NumPy array
    motion = pd.read_csv(csv_path, header=None).values

    if motion.ndim == 1:
        # single row edge case -> reshape to (1, n_features)
        motion = motion.reshape(1, -1)

    if motion.shape[1] > 3:
        motion = motion[:, :3]

    return motion


def explore_basic_info(csv_files, max_print=5):
    """
        Display basic information about a list of CSV motion-capture files.

        Parameters
        ------------
        csv_files : list of str
            Paths to CSV files containing motion data.
        max_print : int, optional
            Maximum number of files to inspect and print information for.
        """

    print(f"Number of CSV files found: {len(csv_files)}\n")
    for path in csv_files[:max_print]:
        motion = load_motion(path)
        print(f"File: {os.path.basename(path)}")
        print(f"  Motion shape: {motion.shape} (time steps x 3D coords)")
        # motion[0] is now the first row as a NumPy array
        print(f"  First row (x, y, z): {motion[0]}")
        print()


def plot_example_motions(motion, title="Example motion"):
    """Plot one motion: 3D plot + x/y/z over time."""
    x = motion[:, 0]
    y = motion[:, 1]
    z = motion[:, 2]

    # 3D plot
    fig = plt.figure(figsize=(10, 4))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(x, y, z, marker="o")
    ax3d.set_title(title)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    # Coordinates vs time
    t = np.arange(len(motion))  # time steps 0,1,...,N-1

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, x, label="x")
    ax2.plot(t, y, label="y")
    ax2.plot(t, z, label="z")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("coordinate value")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def explore_data():
    # find all CSV files in ../data (relative to this script)
    pattern = os.path.join(DATA_DIR, "*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print("No CSV files found in:", DATA_DIR)
        return

    if DEBUG:
        # ------------ Basic info ------------
        # View a summary of your data just select nubmer of files to print
        explore_basic_info(csv_files, max_print=5)

        # ------------ Plot one example motion ------------
        example_path = csv_files[0]
        motion = load_motion(example_path)

        # To visualize one motion, call:
        print(f"Plotting example from: {os.path.basename(example_path)}")
        plot_example_motions(motion, title=os.path.basename(example_path))


if __name__ == "__main__":
    explore_data()