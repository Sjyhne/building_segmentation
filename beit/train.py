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

from data_generator import get_dataset
from beit_seg import BeitSegmentationModel
from metrics import IoU, soft_dice_loss

def train(model, gpu=False):

    training_data = get_dataset("training", data_percentage=0.005)
    print("Len training_data:", len(training_data))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=model.lr)

    # Training loop 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    n_epochs = 1

    model = model.to(device)

    log = {"total_loss": [], "total_iou_5": [], "total_iou_7": [], "total_dice": [], "test_total_loss": [], "test_total_iou_5": [], "test_total_iou_7": [], "test_total_dice": []}

    for _, epoch in tqdm(enumerate(range(n_epochs)), desc="Training loop", total=n_epochs):
        epoch_loss = 0
        epoch_iou_5 = 0
        epoch_iou_7 = 0
        epoch_dice = 0
        for i in range(len(training_data)):
            source, target = training_data[i]

            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            #output = torch.argmax(model(source), dim=3).unsqueeze(3)
            output = model(source)

            oshape = output.shape

            output = output.reshape(oshape[0], oshape[3], oshape[1], oshape[2]).double()

            loss = criterion(output, target)
            # Why?
            loss.backward()
            optimizer.step()

            epoch_iou_5 += round(IoU(output, target, 0.5)/len(training_data))
            epoch_iou_7 += round(IoU(output, target, 0.7)/len(training_data))
            epoch_loss += round(loss.item()/len(training_data))
            epoch_dice += round(soft_dice_loss(output, target)/len(training_data))

        print('[%d]\n Training -> loss: %.3f, iou 0.5: %.3f, iou 0.7: %.3f, dice: %.3f' % (epoch + 1, epoch_loss, epoch_iou_5, epoch_iou_7, epoch_dice))

        log["total_iou_5"].append(epoch_iou_5)
        log["total_iou_7"].append(epoch_iou_7)
        log["total_loss"].append(epoch_loss)
        log["total_dice"].append(epoch_dice)

        test_loss, test_iou_5, test_iou_7, test_dice = evaluate(model, criterion, device)

        log["test_total_iou_5"].append(test_iou_5)
        log["test_total_iou_7"].append(test_iou_7)
        log["test_total_loss"].append(test_loss)
        log["test_total_dice"].append(test_dice)

        print('Test -> loss: %.3f, iou 0.5: %.3f, iou 0.7: %.3f, dice: %.3f' % (test_loss, test_iou_5, test_iou_7, test_dice))


    with open("metrics.json", "w") as file:
        json.dump(log, file)


    print("Finished training")

    return model


def evaluate(model, criterion, device):
    test_data = get_dataset("test")

    test_epoch_loss = 0
    test_epoch_iou_5 = 0
    test_epoch_iou_7 = 0
    test_epoch_dice = 0

    with torch.no_grad():
        for i in range(len(test_data)):
            source, target = test_data[i]
            source, target = source.to(device), target.to(device)

            output = model(source)

            oshape = output.shape

            output = output.reshape(oshape[0], oshape[3], oshape[1], oshape[2]).double()

            loss = criterion(output, target)

            test_epoch_iou_5 += IoU(output, target, 0.5)/len(test_data)
            test_epoch_iou_7 += IoU(output, target, 0.7)/len(test_data)
            test_epoch_loss += loss.item()/len(test_data)
            test_epoch_dice += soft_dice_loss(output, target)/len(test_data)

    return round(test_epoch_loss, 4), round(test_epoch_iou_5, 4), round(test_epoch_iou_7, 4), round(test_epoch_dice, 4)
    ...

if __name__ == "__main__":

    model = BeitSegmentationModel(lr=0.00001, num_classes=2)

    model = model.double()
    model = train(model)

    test_data = get_dataset("test")

    source_images, target_images = test_data[0]
    real_images, real_target_images = test_data.get_images(0)

    output_images = torch.max(model(source_images), dim=3)[0].unsqueeze(3)

    print("output_images.shape", output_images.shape)

    f, axarr = plt.subplots(1, 4)

    axarr[0].imshow(output_images[0].cpu().detach().numpy())
    axarr[1].imshow(real_images[0])
    axarr[2].imshow(real_target_images[0])
    axarr[3].imshow(target_images[0].cpu().detach().numpy().reshape(224, 224, 1))

    plt.savefig("comparison_test.png")