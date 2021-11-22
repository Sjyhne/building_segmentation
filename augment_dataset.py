import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from tqdm import tqdm

import os
import random

import cv2 as cv

def augment_image(aug, image, label):

    if aug == "rt":
        angle = random.choice([45, 145, 225, 315])
        return rotate(image, angle=angle, mode="wrap") * 255, rotate(label, angle=angle, mode="wrap") * 255
    elif aug == "fliplr":
        return np.fliplr(image), np.fliplr(label)
    elif aug == "flipud":
        return np.flipud(image), np.flipud(label)
    elif aug == "blur":
        return gaussian(image, sigma=1, multichannel=True) * 255, label

def augment_dataset(path_to_dataset, augmentation_methods):
    datatypes = ["train", "val", "test"]
    
    # Create new augmented datadir
    augmented_dataset_path = os.path.join("".join(path_to_dataset.split("/")[:-1]), path_to_dataset.split("/")[-1] + "_" + "_".join(sorted(augmentation_methods)))
    print(augmented_dataset_path)
    if not os.path.exists(augmented_dataset_path):
        os.mkdir(augmented_dataset_path)
    else:
        print("PATH ALREADY EXISTS, check whether the datasets exits from an earlier run")
        exit()

    print("Created new dataset directory at:", augmented_dataset_path + "\n")


    for imgtype in ["img_dir", "ann_dir"]:
        for datatype in datatypes:
            os.makedirs(os.path.join(augmented_dataset_path, imgtype, datatype))
    
    print("\nFolder structure complete -- Augmenting images\n")

    for datatype in datatypes:
        current_img_path = os.path.join(path_to_dataset, "img_dir", datatype)
        current_lab_path = os.path.join(path_to_dataset, "ann_dir", datatype)
        print("Augmenting images from:", path_to_dataset, "of type:", datatype)
        img_paths = sorted(os.listdir(current_img_path))
        lab_paths = sorted(os.listdir(current_lab_path))
        for file_index, filename in tqdm(enumerate(img_paths), total=len(img_paths), desc=datatype):
            img = cv.imread(os.path.join(current_img_path, filename))
            lab = cv.imread(os.path.join(current_lab_path, filename))
            cv.imwrite(os.path.join(augmented_dataset_path, "img_dir", datatype, filename), (img).astype(np.uint8))
            cv.imwrite(os.path.join(augmented_dataset_path, "ann_dir", datatype, filename), (lab).astype(np.uint8).max(axis=2).reshape(lab.shape[0], lab.shape[1], 1))
            if lab.mean() >= 0.1:
                for aug in augmentation_methods:
                    augmented_image, augmented_label = augment_image(aug, img, lab)
                    cv.imwrite(os.path.join(augmented_dataset_path, "img_dir", datatype, filename.split(".")[0] + "_" + aug + "." + filename.split(".")[-1]), (augmented_image).astype(np.uint8))
                    cv.imwrite(os.path.join(augmented_dataset_path, "ann_dir", datatype, filename.split(".")[0] + "_" + aug + "." + filename.split(".")[-1]), (augmented_label).astype(np.uint8).max(axis=2).reshape(augmented_label.shape[0], augmented_label.shape[1], 1))


if __name__ == "__main__":
    augment_dataset("datasets/kartai_ksand_manuell_224", ["rt", "fliplr", "flipud", "blur"])