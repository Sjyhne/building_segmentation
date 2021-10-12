import sys
import os.path
from numpy.lib.utils import source
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_generator import get_dataset
from beit_seg import BeitSegmentationModel
from metrics import IoU

def train(model, gpu=False):



    training_data = get_dataset("training", data_percentage=1.0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=model.lr)

    # Training loop

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    n_epochs = 10

    model = model.to(device)

    total_loss = []
    total_iou_5 = []
    total_iou_7 = []
    for _, epoch in tqdm(enumerate(range(n_epochs)), desc="Training loop", total=n_epochs):
        epoch_loss = 0
        epoch_iou_5 = 0
        epoch_iou_7 = 0
        for i in range(len(training_data)):
            source_images, target_images = training_data[i]

            s_images = source_images.double().to(device)
            t_images = target_images.double().to(device)

            optimizer.zero_grad()

            output_images = model(s_images)
            
            output_images = output_images.reshape(output_images.shape[0], output_images.shape[3], output_images.shape[1], output_images.shape[2])

            loss = criterion(output_images, t_images)
            loss.backward()
            optimizer.step()

            running_iou_5 = 0
            running_iou_7 = 0

            for i in range(len(source_images)):
                running_iou_5 += IoU(output_images[i].cpu().detach().numpy(), t_images[i].cpu().detach().numpy(), 0.5)
                running_iou_7 += IoU(output_images[i].cpu().detach().numpy(), t_images[i].cpu().detach().numpy(), 0.7)

            epoch_iou_5 += running_iou_5 / len(output_images)
            epoch_iou_7 += running_iou_7 / len(output_images)
            epoch_loss += loss.item()

        print('[%d] loss: %.3f, iou 0.5: %.3f, iou 0.7: %.3f' % (epoch + 1, epoch_loss/len(training_data), epoch_iou_5/len(training_data), epoch_iou_7/len(training_data)))

    print("Finished training")

    return model


if __name__ == "__main__":

    model = BeitSegmentationModel(lr=0.00001)

    model = model.double()
    model = train(model)

    test_data = get_dataset("test")

    source_images, target_images = test_data[0]

    output_images = model(source_images)

    f, axarr = plt.subplots(1, 3)

    
    axarr[0].imshow(output_images[0].cpu().detach().numpy())
    axarr[1].imshow(source_images[0])
    axarr[2].imshow(target_images[0].cpu().detach().numpy())
    
    plt.show()

