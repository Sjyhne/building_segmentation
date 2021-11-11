import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch

from torch.utils.data import Dataset
from PIL import Image
from transformers import BeitFeatureExtractor
from tqdm import tqdm
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian

import json

import os
import random
import shutil
from sys import argv, platform
from pathlib import Path, PureWindowsPath

import torchvision

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

def get_kartai_image_paths(dataset_name: str, data_type: str):
    main_dir = os.path.join("training_data", "created_datasets")

    dataset_dir = os.path.join(main_dir, dataset_name)

    if data_type == "training":
        paths = json.load(open(os.path.join(dataset_dir, "train_set.json")))
    elif data_type == "test":
        paths = json.load(open(os.path.join(dataset_dir, "test_set.json")))
    elif data_type == "validation":
        paths = json.load(open(os.path.join(dataset_dir, "valid_set.json")))
    else:
        raise RuntimeError(f"Datatype does not exist: {data_type}. Please choose between 'training', 'test', and 'validation'")
    
    source_image_paths, target_image_paths = [], []

    for item in paths:
        source_image_paths.append(item["image"])
        target_image_paths.append(item["label"])
    
    assert len(source_image_paths) == len(target_image_paths)

    return source_image_paths, target_image_paths


def get_dataset(data_type, augmentation_techniques=[], data_percentage=1.0, batch_size=16):
    if data_type == "training":
        return AerialImages(training_data_dir, data_type, data_percentage, augmentation_techniques=augmentation_techniques, batch_size=batch_size, kartai=False)
    elif data_type == "test":
        return AerialImages(test_data_dir, data_type, data_percentage, augmentation_techniques=augmentation_techniques, batch_size=batch_size, kartai=False)
    elif data_type == "validation":
        return AerialImages(validation_data_dir, data_type, data_percentage, augmentation_techniques=augmentation_techniques, batch_size=batch_size, kartai=False)
    else:
        raise RuntimeError("The specified dataset type does not exist. Please choose from the following dataset types: [training, test, validation]")

def get_kartai_dataset(dataset, data_type, augmentation_techniques=[], data_percentage=1.0, batch_size=16):
    if data_type == "training":
        return AerialImages(dataset, data_type, data_percentage, augmentation_techniques=augmentation_techniques, batch_size=batch_size)
    elif data_type == "test":
        return AerialImages(dataset, data_type, data_percentage, augmentation_techniques=augmentation_techniques, batch_size=batch_size)
    elif data_type == "validation":
        return AerialImages(dataset, data_type, data_percentage, augmentation_techniques=augmentation_techniques, batch_size=batch_size)
    else:
        raise RuntimeError("The specified dataset type does not exist. Please choose from the following dataset types: [training, test, validation]")




class AerialImages(Dataset):
    def __init__(self, data_dir, data_type, data_percentage=1.0, augmentation_techniques=[], patch_size=(224, 224), batch_size=16, kartai=True, feature_extractor_model="microsoft/beit-base-patch16-224-pt22k-ft22k"):

        self.augmentation_techniques = augmentation_techniques

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.kartai = kartai
        self.feature_extractor_model = feature_extractor_model
        
        if self.kartai:
            self.source_image_paths, self.target_image_paths = get_kartai_image_paths(data_dir, data_type)
        else:
            self.source_image_paths, self.target_image_paths = get_image_paths(data_dir)

        self.feature_paths, self.label_paths = self._get_features(self.source_image_paths,
                                                                  self.target_image_paths,
                                                                  data_type,
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
            img = cv.imread(image_path) / 255.0
            r += np.mean(img[:, :, 0])
            g += np.mean(img[:, :, 1])
            b += np.mean(img[:, :, 2])
        
        r, g, b = r/len(self.source_image_paths), g/len(self.source_image_paths), b/len(self.source_image_paths)
        return [r, g, b]

    def get_channel_stds(self):

        r_std, g_std, b_std = 0, 0, 0

        for image_path in self.source_image_paths:
            img = cv.imread(image_path) / 255.0
            r_std += np.sum([np.square(np.absolute(r - self.channel_means[0])) for r in img[:, :, 0]])/(img.shape[0] * img.shape[1])
            g_std += np.sum([np.square(np.absolute(g - self.channel_means[1])) for g in img[:, :, 1]])/(img.shape[0] * img.shape[1])
            b_std += np.sum([np.square(np.absolute(b - self.channel_means[2])) for b in img[:, :, 2]])/(img.shape[0] * img.shape[1])

        r_std, g_std, b_std = np.sqrt(r_std/len(self.source_image_paths)), np.sqrt(g_std/len(self.source_image_paths)), np.sqrt(b_std/len(self.source_image_paths))
        return [r_std, g_std, b_std]
    
    def _augment_images(self, aug, images, labels):
        
        augmented_images = []
        augmented_labels = []

        for i in range(len(images)):
            if labels[i].mean() > 0.25:
                if aug == "rt":
                    for angle in [30, 60, 120, 150, 210, 240, 300, 330]:
                        augmented_images.append(rotate(images[i], angle=angle, mode="wrap"))
                        augmented_labels.append(rotate(labels[i], angle=angle, mode="wrap"))
                elif aug == "fliplr":
                    augmented_images.append(np.fliplr(images[i]))
                    augmented_labels.append(np.fliplr(labels[i]))
                elif aug == "flipud":
                    augmented_images.append(np.flipud(images[i]))
                    augmented_labels.append(np.flipud(labels[i]))
                elif aug == "blur":
                    augmented_images.append(gaussian(images[i], sigma=1, multichannel=True))
                    augmented_labels.append(labels[i])

        return augmented_images, augmented_labels


    def _get_features(self,
                      source_image_paths,
                      target_image_paths,
                      data_type,
                      data_percentage
                      ):

        all_features = []
        all_labels = []

        all_features_images = []
        all_labels_images = []

        if len(self.augmentation_techniques) != 0:
            target_dir = Path(os.path.join("features", data_type + "_" + "_".join(sorted(self.augmentation_techniques))))
        else:
            target_dir = Path(os.path.join("features", data_type))


        feature_dir = Path(os.path.join(target_dir), "features")
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

            print("Channel means:", self.channel_means)
            print("Channel standard deviation:", self.channel_stds)

            fe = BeitFeatureExtractor.from_pretrained(
                self.feature_extractor_model,
                image_mean=self.channel_means,
                image_std=self.channel_stds
            )

            for _, i in tqdm(enumerate(range(len(source_image_paths))),
                             total=len(source_image_paths),
                             desc="Creating features and labels"
                             ):
                large_source_image = cv.imread(source_image_paths[i])
                large_target_image = cv.imread(target_image_paths[i])

                source_image_patches, target_image_patches = get_image_patches(large_source_image,
                                                                                     large_target_image,
                                                                                     self.patch_size)

                source_image_patches = source_image_patches / 255.0
                target_image_patches = target_image_patches / 255.0

                source_features = [np.nan_to_num(img.reshape(self.patch_size[0], self.patch_size[1], 3)) for img in source_image_patches]
                
                for aug in self.augmentation_techniques:
                    augmented_images, augmented_labels = self._augment_images(aug, source_features, target_image_patches)
                    all_features.extend(augmented_images)
                    all_labels.extend(augmented_labels)

                    augmented_images, augmented_labels = self._augment_images(aug, source_image_patches, target_image_patches)
                    all_features_images.extend(augmented_images)
                    all_labels_images.extend(augmented_labels)

                source_features = fe(images=source_features, return_tensors="np")["pixel_values"]
                    

                all_features.extend(source_features)
                all_labels.extend(target_image_patches)

                all_features_images.extend(source_image_patches)
                all_labels_images.extend(target_image_patches)

            for _, index in tqdm(enumerate(range(len(all_features))), total=len(all_features), desc="Storing features and labels"):
                np.save(os.path.join(feature_dir, "feature_" + str(index).zfill(len(str(len(all_features)))) + ".npy"), np.asarray(all_features_images[index]))
                np.save(os.path.join(label_dir, "label_" + str(index).zfill(len(str(len(all_features)))) + ".npy"), all_labels_images[index].reshape(self.patch_size[0], self.patch_size[1], 1))
                torch.save(torch.DoubleTensor(all_features[index].reshape(3, self.patch_size[0], self.patch_size[1]).copy()), os.path.join(feature_dir, "feature_" + str(index).zfill(len(str(len(all_features)))) + ".pt"))
                torch.save(torch.DoubleTensor(all_labels[index].reshape(1, self.patch_size[0], self.patch_size[1]).copy()), os.path.join(label_dir, "label_" + str(index).zfill(len(str(len(all_features)))) + ".pt"))
            
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

def get_image_patches(source_image, target_image, patch_size):
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
    target_patches = target_patches.reshape((target_patches.shape[0] * target_patches.shape[1]), target_patches.shape[2], target_patches.shape[3], target_patches.shape[4])

    return source_patches, target_patches

def plot_datapoint(img, lab):
    f, ax = plt.subplots(1, 2, figsize=(16, 16))
    ax[0].imshow(img)
    ax[1].imshow(lab)
    plt.show()

def plot_batch(images, labels, count):
    for idx in range(len(images)):
        plot_datapoint(images[idx], labels[idx])
        if idx + 1 == count:
            break


def create_image_tiles_for_custom_dataset(dataset_name, dataset_type, img_path, label_path, tilesize):
    print("Creating images and labels with tilesize", tilesize, "at", "datasets/" + dataset_name)
    
    main_dir = "datasets"

    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    
    dataset_dir = os.path.join(main_dir, dataset_name)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    img_dir, ann_dir = os.path.join(dataset_dir, "img_dir"), os.path.join(dataset_dir, "ann_dir")

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    if not os.path.exists(ann_dir):
        os.mkdir(ann_dir)
    
    datatype_img_dir, datatype_ann_dir = os.path.join(img_dir, dataset_type), os.path.join(ann_dir, dataset_type)

    if not os.path.exists(datatype_img_dir):
        os.mkdir(datatype_img_dir)
    
    if not os.path.exists(datatype_ann_dir):
        os.mkdir(datatype_ann_dir)
    
    img_files = sorted([os.path.join(img_path, file_path) for file_path in os.listdir(img_path)])
    label_files = sorted([os.path.join(label_path, file_path) for file_path in os.listdir(label_path)])

    for img_file_path, label_file_path in zip(img_files, label_files):
        assert img_file_path.split("/")[-1].split(".")[0] == label_file_path.split("/")[-1].split(".")[0]
    
    print("All images and labels has matching names")

    for img, label in zip(img_files, label_files):
        img_tiles, label_tiles = get_image_patches(cv.imread(img), cv.imread(label), tilesize)
        for i, (img_tile, label_tile) in enumerate(zip(img_tiles, label_tiles)):
            cv.imwrite(os.path.join(datatype_img_dir, img.split("/")[-1].split(".")[0] + "_" + str(i) + "." + img.split("/")[-1].split(".")[1]), img_tile)
            cv.imwrite(os.path.join(datatype_ann_dir, label.split("/")[-1].split(".")[0] + "_" + str(i) + "." + label.split("/")[-1].split(".")[1]), label_tile)
    
    print("Finished writing images to:", dataset_dir)

    


if __name__ == "__main__":

    create_image_tiles_for_custom_dataset("aerial_512", "train", "data/tiff/train", "data/tiff/train_labels", (512, 512))
    create_image_tiles_for_custom_dataset("aerial_512", "test", "data/tiff/test", "data/tiff/test_labels", (512, 512))
    create_image_tiles_for_custom_dataset("aerial_512", "val", "data/tiff/val", "data/tiff/val_labels", (512, 512))

    exit()

    data = get_dataset("training")

    images, labels = data[0]

    print(len(images), len(labels))

    i, l = data.get_images(0)

    plot_batch(i, l, 2)



    """data = get_dataset("test", augmentation_techniques=["blur"], batch_size=10, data_percentage=1.0)

    images, labels = data[0]

    print(images.shape)

    for i in range(len(images)):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(images[i].reshape(224, 224, 3))
        axarr[1].imshow(labels[i].reshape(224, 224, 1))
        plt.show()

    positive_pixel_count = 0
    total_pixel_count = 0

    weights = []

    for img, tar in data:
        positive = tar.sum().numpy()
        count = tar.numel()

        positive_pixel_count += positive
        total_pixel_count += count

        weights.append(positive / count)


    
    print("Positive pixel weight:", positive_pixel_count / total_pixel_count)

    rounded_weights = sorted([round(w, 2) for w in weights])

    rounded_weights_dict = sorted(set(rounded_weights))

    weights, weights_count = [], []

    for weight in rounded_weights_dict:
        w_count = 0
        for w in rounded_weights:
            if weight == w:
                w_count += 1
        weights.append(weight)
        weights_count.append(w_count)

    print(weights)
    print(weights_count)

    plt.bar(weights, weights_count, width=0.1, align="edge", bottom=weights, linewidth=1)
    plt.show()
    """