import matplotlib.pyplot as plt
from data_generator import get_dataset

import numpy as np

def IoU(output_image, target_image, threshold):
    """
        Function for calculating the Intersection of Union for the predicted images.
        * output_image (Predicted image)
        * target_image (Labeled image)
        * 
    """

    if threshold == 0.5:
        output_image = np.round(output_image)
    else:
        output_image = np.where(output_image >= threshold, int(1), int(0))

    intersection = np.logical_and(output_image, target_image)

    union = np.logical_or(output_image, target_image)

    iou = np.sum(intersection) / np.sum(union)

    return iou


def dice_coef():
    ...


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
    