import sys
import os.path
import json

import wandb

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
from metrics import IoULoss, DiceLoss

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def train(model, training_data, test_data, config):

    print("Len training_data:", len(training_data))
    print("Currently using:", config["device"])
    
    target_prod = 0
    target_size = 0
    
    for _, target in training_data:
        target_prod += target.sum()
        target_size += target.numel()
    
    weight = target_prod / target_size
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]).float().to(device))
    optimizer = optim.Adam(params=model.decoder.parameters(), lr=model.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_stepsize"], gamma=config["scheduler_gamma"])

    iou_loss, dice_loss = IoULoss(), DiceLoss()

    n_epochs = config["epochs"]

    model = model.to(device)

    wandb.watch(model, criterion=criterion, log="gradients", log_graph=True)

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

            output = model(source)
            
            target = target.reshape(target.shape[0], target.shape[2], target.shape[3], target.shape[1])

            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()

            output = torch.sigmoid(output)
            
            epoch_iou_5 += iou_loss(output, target).cpu().detach().numpy()
            epoch_loss += loss.item()
            epoch_dice += dice_loss(output, target).cpu().detach().numpy()
            
        scheduler.step()
        
        #lr = lr_scheduler.get_last_lr()[0]
        epoch_iou_5 = epoch_iou_5/len(training_data)
        epoch_loss = epoch_loss/len(training_data)
        epoch_dice = epoch_dice/len(training_data)

        print('[%d]\n Training -> loss: %.3f, iou loss: %.3f, dice loss: %.3f, lr: %f' % (epoch + 1, epoch_loss, epoch_iou_5, epoch_dice, scheduler.get_last_lr()[0]))

        log["total_iou_5"].append(epoch_iou_5)
        log["total_loss"].append(epoch_loss)
        log["total_dice"].append(epoch_dice)

        test_loss, test_iou_5, test_dice = evaluate(model, criterion, test_data, device)
        
        log["test_total_iou_5"].append(test_iou_5)
        log["test_total_loss"].append(test_loss)
        log["test_total_dice"].append(test_dice)
         
        
        wandb.log({"Training loss": epoch_loss})
        wandb.log({"Training iou": epoch_iou_5})
        wandb.log({"Training dice": epoch_dice})
        wandb.log({"Test loss": test_loss})
        wandb.log({"Test iou": test_iou_5})
        wandb.log({"Test dice": test_dice})
        wandb.log({"Learning rate": scheduler.get_last_lr()[0]})
        
        print('Test -> loss: %.3f, iou loss: %.3f, dice loss: %.3f' % (test_loss, test_iou_5, test_dice))


    with open("metrics.json", "w") as file:
        json.dump(log, file)


    print("Finished training")

    return model


def evaluate(model, criterion, test_data, device):

    test_epoch_loss = 0
    test_epoch_iou_5 = 0
    test_epoch_dice = 0

    iou_loss, dice_loss = IoULoss(), DiceLoss()

    with torch.no_grad():
        for i in range(len(test_data)):
            source, target = test_data[i]
            source, target = source.to(device).float(), target.to(device).float()

            output = model(source)

            target = target.reshape(target.shape[0], target.shape[2], target.shape[3], target.shape[1])
            
            loss = criterion(output, target)

            output = torch.sigmoid(output)

            test_epoch_iou_5 += iou_loss(output, target).cpu().detach().numpy()
            test_epoch_loss += loss.item()
            test_epoch_dice += dice_loss(output, target).cpu().detach().numpy()

    return test_epoch_loss/len(test_data), test_epoch_iou_5/len(test_data), test_epoch_dice/len(test_data)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    epochs = 1

    config = wandb.config = {
        "learning_rate": 0.05,
        "epochs": epochs,
        "scheduler_stepsize": epochs//1,
        "scheduler_gamma": 0.4,
        "device": (torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        "num_classes": 1,
        "training": {
            "batch_size": 16,
            "data_percentage": 0.01
        },
        "test": {
            "batch_size": 16,
            "data_percentage": 0.01
        }
    }

    wandb.init(project="beit-kaggle-dataset-uia-server", entity="sjyhne", config=config)

    model = BeitSegmentationModel(lr=wandb.config["learning_rate"], num_classes=config["num_classes"])

    test_data = get_dataset("test", config["test"]["data_percentage"], config["test"]["batch_size"])

    training_data = get_dataset("training", config["training"]["data_percentage"], config["training"]["batch_size"])

    model = train(model, training_data, test_data, config)

    training_source_images, training_target_images = training_data[0]
    training_source_images, training_target_images = training_source_images.to(config["device"]).float(), training_target_images.to(config["device"]).float()

    training_real_images, target_real_images = training_data.get_images(0)

    training_output_images = model(training_source_images)

    source_images, target_images = test_data[0]
    source_images, target_images = source_images.to(config["device"]).float(), target_images.to(config["device"]).float()
    
    real_images, real_target_images = test_data.get_images(0)

    output_images = model(source_images)
    
    output_images = torch.sigmoid(output_images)
    training_output_images = torch.sigmoid(training_output_images)

    f, axarr = plt.subplots(1, 3)
    
    for i in range(len(output_images)):
        
        axarr[0].imshow(output_images[i].cpu().detach().numpy())
        axarr[1].imshow(real_images[i])
        axarr[2].imshow(real_target_images[i])

        plt.savefig(f"comparisons/comparison_test_{i}.png")
    
    f, axarr = plt.subplots(1, 3)
    
    for i in range(len(training_output_images)):
        
        axarr[0].imshow(training_output_images[i].cpu().detach().numpy())
        axarr[1].imshow(training_real_images[i])
        axarr[2].imshow(target_real_images[i])

        plt.savefig(f"comparisons/comparison_training_{i}.png")

