from transformers import BeitModel
from torch import nn
import torch

import matplotlib.pyplot as plt

from PIL import Image
import requests

"""
Picked at https://github.com/920232796/SETR-pytorch/blob/master/SETR/transformer_seg.py
"""

class BranchedDecoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, features=[[256, 256], [256, 256], [512, 256, 128, 64]]):
        super().__init__()
        self.decoder_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0][0], 2, padding=0),
            nn.BatchNorm2d(features[0][0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(features[0][0], features[0][1], 1, padding=0),
            nn.BatchNorm2d(features[0][1]),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        
        self.decoder_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, features[1][0], 2, padding=0),
            nn.BatchNorm2d(features[1][0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(features[0][0], features[0][1], 1, padding=0),
            nn.BatchNorm2d(features[0][1]),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        self.merge_branch = nn.Sequential(
            nn.Conv2d(features[2][0], features[2][1], 2, padding=0),
            nn.BatchNorm2d(features[2][1]),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features[2][1], features[2][2], 2, padding=0),
            nn.BatchNorm2d(features[2][2]),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features[2][2], features[2][3], 2, padding=0),
            nn.BatchNorm2d(features[2][3]),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        self.final_layer = nn.Conv2d(features[2][3], out_channels, 3, padding=0)
        
    
    def forward(self, x):
        x1 = self.decoder_branch_1(x)
        
        x2 = self.decoder_branch_2(x)
        
        x = torch.cat((x1, x2), 1)
        
        x = self.merge_branch(x)
        
        x = self.final_layer(x)
        
        return x

class Decoder2D(nn.Module):

    # TODO: Maybe add dropout? Not sure if it is needed.

    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, features[0], 2, padding=0)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        
        self.conv2 = nn.Conv2d(features[0], features[1], 2, padding=0)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="relu")
        
        self.conv3 = nn.Conv2d(features[1], features[2], 2, padding=0)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, mode="fan_in", nonlinearity="relu")
        
        self.conv4 = nn.Conv2d(features[2], features[3], 2, padding=0)
        torch.nn.init.kaiming_uniform_(self.conv4.weight, mode="fan_in", nonlinearity="relu")
        
        self.decoder_1 = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(features[0]),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            self.conv2,
            nn.BatchNorm2d(features[1]),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            self.conv3,
            nn.BatchNorm2d(features[2]),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.decoder_4 = nn.Sequential(
            self.conv4,
            nn.BatchNorm2d(features[3]),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = (nn.Conv2d(features[-1], out_channels, 3, padding=0))

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x

class BeitSegmentationModel(nn.Module):

    def __init__(
        self,
        lr=0.001,
        momentum=0.9,
        img_size=224,
        patch_dim=16,
        num_channels=3,
        num_classes=2,
        batch_size=16,
        pretrained_model="microsoft/beit-base-patch16-224-pt22k"
    ):

        super(BeitSegmentationModel, self).__init__()

        self.lr = lr
        self.momentum = momentum
        self.img_size = img_size
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size

        # BEiT encoder without segmentation head, can use BeitConfig to
        # alter some of the default values used
        self.beit_base = BeitModel.from_pretrained(pretrained_model)
        
        for param in self.beit_base.parameters():
            param.requires_grad = False
        
        for param in self.beit_base.pooler.parameters():
            param.requires_grad = True

        self.decoder = Decoder2D(self.num_channels, self.num_classes)

    def encode(self, x):
        """
            Our self.encode should just be the BEiT base model calculating
            its hidden state and returning it. It returns the following
            * last_hidden_state
            * pooler_output
            * hidden_states
            * attentions
        """

        encoder_output = self.beit_base(pixel_values=x)[1].reshape(self.batch_size, self.num_channels, self.patch_dim, self.patch_dim)
        
        return encoder_output

    def forward(self, x):

        self.batch_size = len(x)
        
        encoder_pooler = self.encode(x)

        decoder_output = self.decoder(encoder_pooler)

        decoder_output = decoder_output.reshape(self.batch_size, self.img_size, self.img_size, self.num_classes)

        return decoder_output


if __name__ == "__main__":
    p = {
        "img_size": 224,
        "patch_dim": 16,
        "num_channels": 3,
        "num_classes": 2,
        "batch_size": 16,
        "pretrained_model": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    }

    b = BeitSegmentationModel(
            img_size=p["img_size"],
            patch_dim=p["patch_dim"],
            num_channels=p["num_channels"],
            num_classes=p["num_classes"],
            batch_size=p["batch_size"],
            pretrained_model=p["pretrained_model"]
        )

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    url1 = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
    image = Image.open(requests.get(url1, stream=True).raw)
    image2 = Image.open(requests.get(url, stream=True).raw)

    images = [image, image2]

    results = b(images)

    for image in images:
        image.show()

    for image in results.detach().numpy():
        plt.imshow(image)
        plt.show()
