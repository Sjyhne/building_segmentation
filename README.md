# Building Segmentation
Building segmentation in cooperation with the KartAi research project. This repository will explore BEiT and a multi-scale model to be used for building segmentation from orthophotos.

## Dataset

Download [this](https://www.kaggle.com/balraj98/massachusetts-buildings-dataset/download) dataset and place it in a data folder located in the main directory.

Unzip it and run the rename_files.py file. This is due to the labels are spelled tif with only one f, and pillow does not like that.