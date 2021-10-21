import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch

from torch.utils.data import Dataset
from PIL import Image
from transformers import BeitFeatureExtractor
from tqdm import tqdm

import os
import random
import shutil
from sys import platform
from pathlib import Path, PureWindowsPath

main_dir = "./data/tiff"

training_data_dir = [main_dir + "/train", main_dir + "/train_labels"]
test_data_dir = [main_dir + "/test", main_dir + "/test_labels"]
validation_data_dir = [main_dir + "/val", main_dir + "/val_labels"]


def get_image_paths(data_dir):
    source_image_dir, target_image_dir = data_dir

    assert len(os.listdir(source_image_dir)) == len(os.listdir(target_image_dir))
    
    source_image_paths, target_image_paths = [], []

    for file in sorted(os.listdir(source_image_dir)):
        source_image_paths.append(os.path.join(source_image_dir, file))
        target_image_paths.append(os.path.join(target_image_dir, file))
    
    return source_image_paths, target_image_paths


def get_dataset(data_type, data_percentage=1.0, batch_size=16):
    if data_type == "training":
        return AerialImages(training_data_dir, data_type, data_percentage, batch_size=batch_size)
    elif data_type == "test":
        return AerialImages(test_data_dir, data_type, data_percentage, batch_size=batch_size)
    elif data_type == "validation":
        return AerialImages(validation_data_dir, data_type, data_percentage, batch_size=batch_size)
    else:
        raise RuntimeError("The specified dataset type does not exist. Please choose from the following dataset types: [training, test, validation]")


class AerialImages(Dataset):
    def __init__(self, data_dir, data_type, data_percentage=1.0, transform=None, patch_size=(224, 224), batch_size=16, feature_extractor_model="microsoft/beit-base-patch16-224-pt22k-ft22k"):
        self.transform = transform

        self.patch_size = patch_size
        self.batch_size = batch_size

        self.source_image_paths, self.target_image_paths = get_image_paths(data_dir)

        self.feature_paths, self.label_paths = self._get_features(self.source_image_paths,
                                                                  self.target_image_paths,
                                                                  data_type,
                                                                  feature_extractor_model,
                                                                  data_percentage)

        # Todo: Create bacthes
        self.batch_feature_paths = []
        self.batch_label_paths = []

        div, mod = divmod(len(self.feature_paths), self.batch_size)
        if mod != 0:
            div += 1

        for i in range(div):
            self.batch_feature_paths.append(self.feature_paths[i * self.batch_size:(i + 1) * self.batch_size])
            self.batch_label_paths.append(self.label_paths[i * self.batch_size:(i + 1)*self.batch_size])

    def get_channel_means(self):

        r, g, b = 0, 0, 0
        
        for image_path in self.source_image_paths:
            img = cv.imread(image_path)/255
            r += np.mean(img[:, :, 0])
            g += np.mean(img[:, :, 1])
            b += np.mean(img[:, :, 2])
        
        r, g, b = r/len(self.source_image_paths), g/len(self.source_image_paths), b/len(self.source_image_paths)
        return [r, g, b]

    def get_channel_stds(self):

        r_std, g_std, b_std = 0, 0, 0

        for image_path in self.source_image_paths:
            img = cv.imread(image_path)/255
            r_std += np.sum([np.square(np.absolute(r - self.channel_means[0])) for r in img[:, :, 0]])/(img.shape[0] * img.shape[1])
            g_std += np.sum([np.square(np.absolute(g - self.channel_means[1])) for g in img[:, :, 1]])/(img.shape[0] * img.shape[1])
            b_std += np.sum([np.square(np.absolute(b - self.channel_means[2])) for b in img[:, :, 2]])/(img.shape[0] * img.shape[1])

        r_std, g_std, b_std = np.sqrt(r_std/len(self.source_image_paths)), np.sqrt(g_std/len(self.source_image_paths)), np.sqrt(b_std/len(self.source_image_paths))
        return [r_std, g_std, b_std]

    def _get_features(self,
                      source_image_paths,
                      target_image_paths,
                      data_type,
                      feature_extractor_model,
                      data_percentage
                      ):

        all_features = []
        all_labels = []

        all_features_images = []
        all_labels_images = []

        target_dir = Path(os.path.join("features", feature_extractor_model, data_type))
        feature_dir = Path(os.path.join(target_dir, "features"))
        label_dir = Path(os.path.join(target_dir, "labels"))

        if platform == "win32":
            target_dir = PureWindowsPath(target_dir)
            feature_dir = PureWindowsPath(feature_dir)
            label_dir = PureWindowsPath(label_dir)
        
        # if not "./features/model.../train"
        if not os.path.exists(os.path.join(target_dir)):
            print("Did not find:", target_dir, "... Creating necessary dirs")
            os.mkdir(target_dir)
            os.mkdir(feature_dir)
            os.mkdir(label_dir)

            self.channel_means = self.get_channel_means()
            self.channel_stds = self.get_channel_stds()

            fe = BeitFeatureExtractor.from_pretrained(
                feature_extractor_model,
                image_mean=self.channel_means,
                image_std=self.channel_stds
            )

            for _, i in tqdm(enumerate(range(len(source_image_paths))),
                             total=len(source_image_paths),
                             desc="Creating features and labels"
                             ):
                large_source_image = cv.imread(source_image_paths[i])
                large_target_image = cv.imread(target_image_paths[i]) / 255.0

                source_image_patches, target_image_patches = self._get_image_patches(large_source_image,
                                                                                     large_target_image,
                                                                                     self.patch_size)

                source_features = fe(images=source_image_patches, return_tensors="pt")["pixel_values"]

                all_features.extend(source_features)
                all_labels.extend(target_image_patches)

                all_features_images.extend(source_image_patches)
                all_labels_images.extend(target_image_patches)
            
            # Releasing memory used by fe
            del fe

            for _, index in tqdm(enumerate(range(len(all_features))), total=len(all_features), desc="Storing features and labels"):
                np.save(os.path.join(feature_dir, "feature_" + str(index).zfill(len(str(len(all_features)))) + ".npy"), np.asarray(all_features_images[index]) / 255.0)
                np.save(os.path.join(label_dir, "label_" + str(index).zfill(len(str(len(all_features)))) + ".npy"), all_labels_images[index].reshape(self.patch_size[0], self.patch_size[1], 1))
                torch.save(all_features[index].double(), os.path.join(feature_dir, "feature_" + str(index).zfill(len(str(len(all_features)))) + ".pt"))
                torch.save(torch.from_numpy(all_labels[index]).double(), os.path.join(label_dir, "label_" + str(index).zfill(len(str(len(all_features)))) + ".pt"))
            
            print("Finished creating features and storing labels and features")

        feature_paths = []
        for path in sorted(os.listdir(feature_dir)):
            if path.split(".")[-1] == "pt":
                feature_paths.append(os.path.join(feature_dir, path))

        label_paths = []
        for path in sorted(os.listdir(label_dir)):
            if path.split(".")[-1] == "pt":
                label_paths.append(os.path.join(label_dir, path))

        combined_paths = list(zip(feature_paths, label_paths))

        random.shuffle(combined_paths)

        feature_paths, label_paths = zip(*combined_paths)

        feature_paths = feature_paths[:int(len(feature_paths) * data_percentage)]
        label_paths = label_paths[:int(len(label_paths) * data_percentage)]

        return feature_paths, label_paths



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
        x_patches = img_w // patch_w

        if img_h % patch_h != 0:
            y_patches += 1
        
        if img_w % patch_w != 0:
            x_patches += 1

        # y, x, patch_size, patch_size, channels ---> [7, 7, 224, 224, 3/1]
        # a = [2, 2, 2]
        source_patches = np.zeros((y_patches, x_patches, patch_size[0], patch_size[1], 3), dtype=np.uint8)
        target_patches = np.zeros((y_patches, x_patches, patch_size[0], patch_size[1], 1), dtype=np.uint8)


        for y in range(y_patches):
            for x in range(x_patches):
                source_patch = np.full((patch_size[0], patch_size[1], 3), (255, 102, 255), dtype=np.uint8)
                target_patch = np.zeros((patch_size[0], patch_size[1], 1), dtype=np.uint8)
                #                               0 * 224 = 0 : (0 + 1) * 224 = 224
                temp_source_patch = source_image[y * patch_size[0]:(y + 1) * patch_size[0], x * patch_size[1]:(x + 1) * patch_size[1]]
                temp_target_patch = target_image[y * patch_size[0]:(y + 1) * patch_size[0], x * patch_size[1]:(x + 1) * patch_size[1]]

                temp_target_patch = np.amax(temp_target_patch, axis=2).reshape(temp_target_patch.shape[0], temp_target_patch.shape[1], 1)

                if temp_source_patch.shape[0] != patch_size[0] or temp_source_patch.shape[1] != patch_size[1]:
                    source_patch[:temp_source_patch.shape[0], :temp_source_patch.shape[1]] = temp_source_patch
                    target_patch[:temp_target_patch.shape[0], :temp_target_patch.shape[1]] = temp_target_patch
                else:
                    source_patch = temp_source_patch
                    target_patch = temp_target_patch

                source_patches[y, x] = source_patch
                target_patches[y, x] = target_patch

        source_patches = source_patches.reshape((source_patches.shape[0] * source_patches.shape[1]), source_patches.shape[2], source_patches.shape[3], source_patches.shape[4])
        target_patches = target_patches.reshape((target_patches.shape[0] * target_patches.shape[1]), target_patches.shape[4], target_patches.shape[2], target_patches.shape[3])

        source_patches = [Image.fromarray(np.uint8(arr)) for arr in source_patches]

        return source_patches, target_patches


    def __len__(self):
        return len(self.batch_feature_paths)

    def __getitem__(self, idx):

        features = torch.empty((len(self.batch_feature_paths[idx]), 3, self.patch_size[0], self.patch_size[1]), dtype=torch.double)
        labels = torch.empty((len(self.batch_label_paths[idx]), 1, self.patch_size[0], self.patch_size[1]), dtype=torch.double)

        for i, f in enumerate(self.batch_feature_paths[idx]):
            feature = torch.load(f)
            features[i] = feature

        for i, l in enumerate(self.batch_label_paths[idx]):
            label = torch.load(l)
            labels[i] = label

        return features, labels

    def get_images(self, idx):
        
        feature_images = np.zeros((len(self.batch_feature_paths[idx]), self.patch_size[0], self.patch_size[1], 3))
        label_images = np.zeros((len(self.batch_label_paths[idx]), self.patch_size[0], self.patch_size[1], 1))

        for i, f in enumerate(self.batch_feature_paths[idx]):
            new_path = ".".join(f.split(".")[:-1]) + ".npy"
            feature_image = np.load(new_path)
            feature_images[i] = feature_image
        
        for i, f in enumerate(self.batch_label_paths[idx]):
            new_path = ".".join(f.split(".")[:-1]) + ".npy"
            label_image = np.load(new_path)
            label_images[i] = label_image
        
        return feature_images, label_images



if __name__ == "__main__":

    data = get_dataset("validation")
    """
    source_tensor, target_tensor = data[0]
    source_img, target_img = data.get_images(0)

    f, axarr = plt.subplots(1, 2)
    
    print(source_img[0])
    print(target_img[0])
    print(source_img.shape)
    print(target_img.shape)

    axarr[0].imshow(source_img[0].reshape(224, 224, 3))
    axarr[1].imshow(target_img[0].reshape(224, 224, 1))

    plt.show()
    """