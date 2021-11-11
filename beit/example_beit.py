import os
from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import matplotlib.pyplot as plt
import numpy as np

import requests

import torchvision.transforms as transforms

import torch

from data_generator import get_dataset

import cv2

from timm.models import create_model
from timm.models import list_models

"""
https://www.kaggle.com/piantic/vision-transformer-vit-visualize-attention-map
"""




#pretrained = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k", output_attentions=True)
#base = BeitModel.from_pretrained("microsoft/beit-base-patch16-224", output_attentions=True)
#finetuned = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k", output_attentions=True)

#models = [pretrained, base, finetuned]

print(list_models())

model = create_model(
    model_name="beit_base_patch16_224_8k_vocab",
    pretrained=False,
    drop_path_rate=0.1,
    drop_block_rate=None,
    )

checkpoint = torch.load("model_checkpoints/checkpoint-1299.pth", map_location=torch.device("cpu"))

print(checkpoint)

model.load_state_dict(checkpoint['model'])

data = get_dataset("test", data_percentage=0.01, batch_size=8)

image, label = data[0]
img, _ = data.get_images(0)

plt.imshow(image[0].permute(1, 2, 0))
plt.show()

trans = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])

t_img = trans(image[0])

b_img = feature_extractor(images=image[0], return_tensors="pt")

print(b_img)

plt.imshow(t_img.permute(1, 2, 0))
plt.show()


print(b_img["pixel_values"].shape)

plt.imshow(b_img["pixel_values"].permute(0, 2, 3, 1).squeeze(0))
plt.show()

def get_attention_info(img, model):

    img = img.float()

    o = model(pixel_values=img.unsqueeze(0))

    att_mat = o["attentions"]

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    grid_size = int(np.sqrt(aug_att_mat.shape[0]))
    
    return joint_attentions, grid_size

def get_attention_map(image, org_img, model, get_mask=False):

    image = image.float()

    o = model(pixel_values=image.unsqueeze(0))

    att_mat = o["attentions"]

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), org_img.shape[:2])
    else:        
        mask = mask / mask.max()
        mask = cv2.resize(mask, org_img.shape[:2])
        mask = mask[..., np.newaxis]
        result = (mask * org_img.reshape(224, 224, 3))
    

    return result

def plot_attention_map(original_img, model_id, att_map, mask):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title(f'Original: {model_id}')
    ax2.set_title(f'Attention Map Last Layer: {model_id}')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    if mask:
        plt.savefig("model_" + str(model_id) + "_mask_attention.jpeg")
    else:
        plt.savefig("model_" + str(model_id) + "_no_mask_attention.jpeg")
    plt.show()

#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#url1 = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
#image1 = Image.open(requests.get(url, stream=True).raw)



#att_mat = get_attention_map(image, img[0])

#plot_attention_map(img[0], att_mat)
"""
joint_att1, grid_size1 = get_attention_info(image[0], pretrained)

for i, v in enumerate(joint_att1):
    v = joint_att1[-1]
    mask = v[0, 1:].reshape(grid_size1, grid_size1).detach().numpy()
    print(mask.shape)
    mask = cv2.resize(mask / mask.max(), image[0].shape[:2])[..., np.newaxis]
    result = (mask * image[0].numpy().reshape(224, 224, 3)).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map_%d Layer' % (i+1))
    _ = ax1.imshow(image[0])
    _ = ax2.imshow(result)
"""
"""for i, model in enumerate(models):
    att_mat_no_mask = get_attention_map(image[0], img[0], model, get_mask=False)
    att_mat_mask = get_attention_map(image[0], img[0], model, get_mask=True)
    plot_attention_map(img[0], i, att_mat_no_mask, mask=False)
    plot_attention_map(img[0], i, att_mat_mask, mask=True)"""