import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from data_generator import get_dataset
from beit_seg import BeitSegmentationModel

def train(model):

    training_data = get_dataset("training")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=model.lr)

    # Training loop

    n_epochs = 1

    running_loss = 0
    for _, epoch in tqdm(enumerate(range(n_epochs)), desc="Training loop", total=n_epochs):
        for i, data in enumerate(training_data):
            source_images, target_images = data

            source_images = source_images
            target_images = torch.tensor(target_images).float()

            optimizer.zero_grad()

            output_images = model(source_images)
            loss = criterion(output_images, target_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0

    print("Finished training")


if __name__ == "__main__":

    model = BeitSegmentationModel(lr=0.00001)

    train(model)