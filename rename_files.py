import os
from PIL import Image
import numpy as np

train_label_dir = "./data/tiff/train_labels"
test_label_dir = "./data/tiff/test_labels"
validation_label_dir = "./data/tiff/val_labels"


dirs = [train_label_dir, test_label_dir, validation_label_dir]

for dir in dirs:
    for file in os.listdir(dir):
        os.rename(os.path.join(dir, file), os.path.join(dir, file + "f"))


'''
import matplotlib.pyplot as plt

if __name__ == "__main__":

    paths = os.listdir("./pretraining_data/224_train")


    for i, path in enumerate(paths):
        a = np.load("./pretraining_data/224_train/" + path)
        plt.imsave(f'feature_{i}.jpeg', a)
'''
