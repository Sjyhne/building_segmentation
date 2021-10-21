import sys
import os.path
import json

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data_generator import get_dataset
from beit_seg import BeitSegmentationModel
from metrics import IoU, soft_dice_loss

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def train(model, gpu=False):

    training_data = get_dataset("training", data_percentage=0.4, batch_size=64)
    print("Len training_data:", len(training_data))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    
    #target_size, target_sum = 0, 0

    #for _, target in training_data:
    #    target_size += np.prod(target.shape)
    #    target_sum += target.sum()
    
    #class_weights = torch.tensor([1 - (target_size/target_sum), target_size/target_sum]).float().to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=model.decoder.parameters(), lr=model.lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)

    n_epochs = 200

    model = model.to(device)

    log = {"total_loss": [], "total_iou_5": [], "total_dice": [], "test_total_loss": [], "test_total_iou_5": [], "test_total_dice": []}

    for _, epoch in tqdm(enumerate(range(n_epochs)), desc="Training loop", total=n_epochs):
        epoch_loss = 0
        epoch_iou_5 = 0
        epoch_dice = 0
        for i in range(len(training_data)):
            source, target = training_data[i]

            source = source.to(device).float()
            target = target.to(device).float()

            optimizer.zero_grad()

            output = torch.sigmoid(model(source))
            
            output = output.reshape(output.shape[0], output.shape[3], output.shape[1], output.shape[2])
            target = target.reshape(target.shape[0], target.shape[1], target.shape[2], target.shape[3])
            
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()

            output = output.reshape(output.shape[0], output.shape[2], output.shape[3], output.shape[1])
            target = target.reshape(target.shape[0], target.shape[2], target.shape[3], target.shape[1])
            
            epoch_iou_5 += round(IoU(output, target, 0.5), 4)
            epoch_loss += round(loss.item(), 4)
            epoch_dice += round(soft_dice_loss(output, target), 4)
            
            #lr_scheduler.step(epoch + i / len(training_data))
        
        scheduler.step()
        
        #lr = lr_scheduler.get_last_lr()[0]
        epoch_iou_5 = epoch_iou_5/len(training_data)
        epoch_loss = epoch_loss/len(training_data)
        epoch_dice = epoch_dice/len(training_data)

        print('[%d]\n Training -> loss: %.3f, iou 0.5: %.3f, dice: %.3f, lr: %.9f' % (epoch + 1, epoch_loss, epoch_iou_5, epoch_dice, scheduler.get_last_lr()[0]))

        log["total_iou_5"].append(epoch_iou_5)
        log["total_loss"].append(epoch_loss)
        log["total_dice"].append(epoch_dice)

        test_loss, test_iou_5, test_dice = evaluate(model, criterion, device)
        
        log["test_total_iou_5"].append(test_iou_5)
        log["test_total_loss"].append(test_loss)
        log["test_total_dice"].append(test_dice)
        
        print('Test -> loss: %.3f, iou 0.5: %.3f, dice: %.3f' % (test_loss, test_iou_5, test_dice))


    with open("metrics.json", "w") as file:
        json.dump(log, file)


    print("Finished training")

    return model


def evaluate(model, criterion, device):
    test_data = get_dataset("test")

    test_epoch_loss = 0
    test_epoch_iou_5 = 0
    test_epoch_dice = 0

    with torch.no_grad():
        for i in range(len(test_data)):
            source, target = test_data[i]
            source, target = source.to(device).float(), target.to(device).float()

            output = torch.sigmoid(model(source))

            output = output.reshape(output.shape[0], output.shape[3], output.shape[1], output.shape[2])
            target = target.reshape(target.shape[0], target.shape[1], target.shape[2], target.shape[3])

            loss = criterion(output, target)

            output = output.reshape(output.shape[0], output.shape[2], output.shape[3], output.shape[1])
            target = target.reshape(target.shape[0], target.shape[2], target.shape[3], target.shape[1])
            
            test_epoch_iou_5 += IoU(output, target, 0.5)
            test_epoch_loss += loss.item()
            test_epoch_dice += soft_dice_loss(output, target)

    return round(test_epoch_loss/len(test_data), 4), round(test_epoch_iou_5/len(test_data), 4), round(test_epoch_dice/len(test_data), 4)


if __name__ == "__main__":

    model = BeitSegmentationModel(lr=0.04, num_classes=1)

    model = model
    model = train(model)

    test_data = get_dataset("test")

    source_images, target_images = test_data[0]
    source_images, target_images = source_images.to("cuda:0").float(), target_images.to("cuda:0").long()
    
    real_images, real_target_images = test_data.get_images(0)

    output_images = model(source_images)

    f, axarr = plt.subplots(1, 3)
    
    for i in range(len(output_images)):
        
        axarr[0].imshow(output_images[i].cpu().detach().numpy())
        axarr[1].imshow(real_images[i])
        axarr[2].imshow(real_target_images[i])

        plt.savefig(f"comparisons/comparison_test_{i}.png")
