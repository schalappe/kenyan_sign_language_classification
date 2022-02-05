# -*- coding: utf-8 -*-
"""
    Script used to create a h5py file that contains the data for training
"""
from os.path import join

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.config import CLASS_NAMES, FEATURES_PATH, INPUT_CSV, TRAIN_PATH
from src.data import DatasetWriter
from src.processors import aspect_resize, load_image

# extracts from csv file
train_labels = read_csv(INPUT_CSV, header="infer")

# set
train_set = [join(TRAIN_PATH, image + ".jpg") for image in train_labels.img_IDS]
label_set = list(train_labels.Label)

# encode
le = LabelEncoder()
le.fit(CLASS_NAMES)
label_set = le.transform(label_set)

# split
x_train, x_val, y_train, y_val = train_test_split(
    train_set, label_set, test_size=0.2, random_state=1337, stratify=label_set
)

# path output
datasets = [
    ("train", x_train, y_train, join(FEATURES_PATH, "train_set.hdf5")),
    ("test", x_val, y_val, join(FEATURES_PATH, "test_set.hdf5")),
]

# loop over dataset and write
for (dType, paths, labels, output) in datasets:
    print(f"[INFO] building {output} ...")
    writer = DatasetWriter((len(x_train), 256, 256, 3), output)

    # loop over images
    for (path, label) in tqdm(
        zip(paths, labels),
        ascii=True,
        desc="Loop over images: ",
        ncols=88,
        colour="green",
    ):
        # load image
        image = load_image(path)

        # resize image
        image = aspect_resize(image, height=256, width=256)

        # add image and label to HDF5 dataset
        writer.add([image], [label])

    # close the HDF5 writer
    writer.close()
