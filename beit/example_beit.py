from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image

import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url1 = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image1 = Image.open(requests.get(url, stream=True).raw)

feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k', output_attentions=True)
images = [image, image1]

inputs = feature_extractor(images=images, return_tensors="pt")

inputs2 = feature_extractor(images=images, return_tensors="pt")


outputs = model(**inputs)

print(outputs.keys())

last_hidden_states = outputs.last_hidden_state

print(outputs.attentions)
for att in outputs.attentions:
    print(att.shape)
print(len(outputs.attentions))