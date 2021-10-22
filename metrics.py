import matplotlib.pyplot as plt
from data_generator import get_dataset

import numpy as np
import torch
import torch.nn as nn

"""
    Losses found at: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):     
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

if __name__ == "__main__":

    train_data = get_dataset("training", 0.01, batch_size=4)

    test = torch.tensor(np.random.standard_normal((224, 224, 2)))

    test = torch.softmax(test, dim=2)[:, :, 1]
    
    print(test[0, 0])

    s_images, t_images = train_data[0]

    print(type(test))

    print(s_images.shape)
    print(t_images.shape)

    dice_loss = DiceLoss()
    iou_loss = IoULoss()

    t = 0.9

    t_images = t_images.reshape(t_images.shape[0], t_images.shape[2], t_images.shape[3], t_images.shape[1])

    r = iou_loss(t_images[0], test)

    dice = dice_loss(t_images[0], test)

    print("Dice loss:", dice)

    print("IoU Loss:", r)

    f, axarr = plt.subplots(1, 2)

    axarr[0].imshow(t_images[0])
    axarr[1].imshow(test)

    plt.show()

