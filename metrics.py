import matplotlib.pyplot as plt
from data_generator import get_dataset

import numpy as np
import torch

from utils import UnNormalize

def IoU(output_image, target_image, threshold):
    """
        Function for calculating the Intersection of Union for the predicted images.
        * output_image (Predicted image)
        * target_image (Labeled image)
        * 
    """

    output_image = output_image.cpu().detach().numpy()
    target_image = target_image.cpu().detach().numpy()

    if threshold == 0.5:
        output_image = np.round(output_image)
    else:
        output_image = np.where(output_image >= threshold, int(1), int(0))

    intersection = np.logical_and(output_image, target_image)

    union = np.logical_or(output_image, target_image)

    return np.sum(intersection) / np.sum(union)


def soft_dice_loss(output_image, target_image, epsilon=1e-6):

    output_image, target_image = output_image.cpu().detach().numpy(), target_image.cpu().detach().numpy()

    axes = tuple(range(1, len(output_image.shape) - 1))
    numerator = 2. * np.sum(output_image * target_image, axes)
    denominator = np.sum(np.square(output_image) + np.square(target_image), axes)

    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon))


if __name__ == "__main__":
    
    train_data = get_dataset("training")

    test = np.random.standard_normal((224, 224, 1))

    s_images, t_images = train_data[0]

    t = 0.9

    r = IoU(test, t_images[0], t)

    print("IoU:", r)

    f, axarr = plt.subplots(1, 2)

    test = np.where(test >= t, int(1), int(0))

    axarr[0].imshow(t_images[0])
    axarr[1].imshow(test)

    plt.show()
    