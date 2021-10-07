import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_generator import get_dataset
from beit_seg import BeitSegmentationModel
from metrics import IoU

def train(model):

    training_data = get_dataset("training", data_percentage=0.3)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=model.lr)

    # Training loop

    n_epochs = 3

    running_loss = 0
    for _, epoch in tqdm(enumerate(range(n_epochs)), desc="Training loop", total=n_epochs):
        for i in range(len(training_data)):
            source_images, target_images = training_data[i]

            source_images = source_images
            target_images = torch.tensor(target_images).float()

            optimizer.zero_grad()

            output_images = model(source_images)

            loss = criterion(output_images, target_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            running_iou_5 = 0
            running_iou_7 = 0

            for i in range(len(source_images)):
                running_iou_5 += IoU(output_images[i].detach().numpy(), target_images[i].detach().numpy(), 0.5)
                running_iou_7 += IoU(output_images[i].detach().numpy(), target_images[i].detach().numpy(), 0.7)

            iou_5 = running_iou_5 / len(output_images)
            iou_7 = running_iou_7 / len(output_images)

            print('[%d, %5d] loss: %.3f, iou 0.5: %.3f, iou 0.7: %.3f' % (epoch + 1, i + 1, running_loss, iou_5, iou_7))
            running_loss = 0

    print("Finished training")

    return model


if __name__ == "__main__":

    model = BeitSegmentationModel(lr=0.00001)

    model = train(model)

    test_data = get_dataset("test")

    source_images, target_images = test_data[0]

    output_images = model(source_images)

    f, axarr = plt.subplots(1, 3)

    
    axarr[0].imshow(output_images[0].detach().numpy())
    axarr[1].imshow(source_images[0])
    axarr[2].imshow(target_images[0])
    
    plt.show()

