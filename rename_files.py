import os

train_label_dir = "./data/tiff/train_labels"
test_label_dir = "./data/tiff/test_labels"
validation_label_dir = "./data/tiff/val_labels"


dirs = [train_label_dir, test_label_dir, validation_label_dir]

for dir in dirs:
    for file in os.listdir(dir):
        os.rename(os.path.join(dir, file), os.path.join(dir, file + "f"))