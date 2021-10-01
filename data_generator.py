import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.utils import source
from torch.utils.data import Dataset

from sklearn.feature_extraction.image import extract_patches_2d

import os
import cv2 as cv

main_dir = "./data/tiff"

training_data_dir = [main_dir + "/train", main_dir + "/train_labels"]
test_data_dir = [main_dir + "/test", main_dir + "test_labels"]
validation_data_dir = [main_dir + "/val", main_dir + "/val_labels"]

def get_image_paths(data_dir):
    source_image_dir, target_image_dir = data_dir

    assert len(os.listdir(source_image_dir)) == len(os.listdir(target_image_dir))
    
    print("Number of images:", len(os.listdir(source_image_dir)))

    source_image_paths, target_image_paths = [], []

    for file in sorted(os.listdir(source_image_dir)):
        source_image_paths.append(os.path.join(source_image_dir, file))
        target_image_paths.append(os.path.join(target_image_dir, file))
    
    return source_image_paths, target_image_paths


class AerialImages(Dataset):
    def __init__(self, data_dir, transform=None):
        self.source_image_paths, self.target_image_paths = get_image_paths(data_dir)

        self.transform = transform

    def _get_image_patches(self, source_image, target_image, patch_size):
        """
            This function will get patches to the correct image size
            for the model. I.e. BEiT is pretrained om 224x224 images.
            Therefore we have to create patches of size 224x224 
            for each of the 1500x1500 images in the dataset.
        """

        img_h, img_w = source_image.shape[0], source_image.shape[1]
        patch_h, patch_w = patch_size[0], patch_size[1]

        y_patches = img_h // patch_h
        x_patches = img_w // patch_h

        if img_h % patch_h != 0:
            y_patches += 1
        
        if img_w % patch_w != 0:
            x_patches += 1

        # Batchsize, patch_size, patch_size, channels
        source_patches = np.zeros((y_patches, x_patches, patch_size[0], patch_size[1], 3), dtype=np.int64)
        target_patches = np.zeros((y_patches, x_patches, patch_size[0], patch_size[1], 3), dtype=np.int64)

        for y in range(y_patches):
            for x in range(x_patches):
                source_patch = np.zeros((patch_size[0], patch_size[1], 3))
                target_patch = np.zeros((patch_size[0], patch_size[1], 3))

                temp_source_patch = source_image[y * patch_size[0]:(y + 1) * patch_size[0], x * patch_size[1]:(x + 1) * patch_size[1]]
                temp_target_patch = target_image[y * patch_size[0]:(y + 1) * patch_size[0], x * patch_size[1]:(x + 1) * patch_size[1]]

                if temp_source_patch.shape[0] != patch_size[0] or temp_source_patch.shape[1] != patch_size[1]:
                    source_patch[:temp_source_patch.shape[0], :temp_source_patch.shape[1]] = temp_source_patch
                    target_patch[:temp_target_patch.shape[0], :temp_target_patch.shape[1]] = temp_target_patch
                else:
                    source_patch = temp_source_patch
                    target_patch = temp_target_patch

                source_patches[y, x] = source_patch
                target_patches[y, x] = target_patch


        return source_patches, target_patches


    def __len__(self):
        return len(self.source_image_paths)

    def __getitem__(self, idx):
        
        source_image = cv.imread(self.source_image_paths[idx])
        target_image = cv.imread(self.target_image_paths[idx])

        print(self.source_image_paths[idx])
        print(self.target_image_paths[idx])

        print(target_image)
        print(source_image)

        source_patches, target_patches = self._get_image_patches(source_image, target_image, (224, 224))

        return source_patches, target_patches
        


if __name__ == "__main__":

    data = AerialImages(training_data_dir)

    source_img, target_img = data[0]

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(source_img[0, -1])
    axarr[1].imshow(target_img[0, -1])
    plt.show()
    #image = cv.imread("./data/tiff/train/22678915_15.tiff")

    #print(image.shape)

    #patches = extract_patches_2d(image, (224, 224))