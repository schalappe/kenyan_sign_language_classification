# -*- coding: utf-8 -*-
"""
    Script used to split data into train and test set
"""
import shutil
from os.path import join

from pandas import read_csv
from tqdm import tqdm

from src.config import IMAGES_PATH, INPUT_CSV, TRAIN_PATH

# extracts from csv file
train_labels = read_csv(INPUT_CSV, header="infer")

# loop over label
for image_name in tqdm(
    train_labels["img_IDS"], ncols=100, desc="Move image for training: "
):
    shutil.move(join(IMAGES_PATH, image_name) + '.jpg', TRAIN_PATH)

print("Successfully complete")
