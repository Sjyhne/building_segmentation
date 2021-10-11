from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image

import requests
import time



l = range(123)

batch = 16

d = len(l)//16
print(len(l)%16)
if len(l)%16 != 0:
    d += 1

t = []
for i in range(d):
    t.append(l[i*batch:(i+1)*batch])

for v in t:
    print(len(v))


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url1 = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image1 = Image.open(requests.get(url, stream=True).raw)

feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

images = [image, image1]
start = time.time()
inputs = feature_extractor(images=images, return_tensors="pt")
print(inputs["pixel_values"].size())
end = time.time()

inputs2 = feature_extractor(images=images, return_tensors="pt")


print("elapsed:", end - start)

start = time.time()
outputs = model(**inputs)
end = time.time()

print("elapsed:", end - start)

last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)