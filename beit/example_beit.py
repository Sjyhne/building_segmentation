from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import requests

import torchvision

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url1 = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image1 = Image.open(requests.get(url, stream=True).raw)



transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
normalized_img = transform(image)

plt.imshow(normalized_img)