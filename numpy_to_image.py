import os
from posixpath import supports_unicode_filenames
import sys

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # Check if source and target dir exists

    if len(sys.argv) < 3:
        raise RuntimeError(f"Missing arguments for script '{sys.argv[0]}'")

    source_directory = sys.argv[1]
    target_directory = sys.argv[2]

    print(source_directory)
    print(target_directory)

    if not os.path.exists(source_directory):
        raise RuntimeError(f"Source directory '{source_directory}' does not exist")
    if not os.path.exists(target_directory):
        raise RuntimeError(f"Target directory '{target_directory}' does not exist")

    paths = os.listdir(source_directory)

    print("Removing contents in", target_directory)
    for path in os.listdir(target_directory):
        os.remove(os.path.join(target_directory, path))

    assert len(os.listdir(target_directory)) == 0

    print("Filling target directory with images")
    for i, path in enumerate(paths):
        if path.split(".")[1] == "npy":
            a = np.load(os.path.join(source_directory, path))
            plt.imsave(os.path.join(target_directory, path.split(".")[0] + ".jpeg"), a)