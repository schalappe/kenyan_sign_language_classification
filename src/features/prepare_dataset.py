# -*- coding: utf-8 -*-
"""
    Script used to create a TFRecordDataset that contains the data for training
"""
from os.path import join

from src.config import FEATURES_PATH
from src.data import split_and_load_data, write_images_to_tfr

print("[INFO]: Split and Load dataset")
input_train, input_val, output_train, output_val = split_and_load_data()

print("[INFO]: Store Train TFRecord")
write_images_to_tfr(
    images=input_train,
    labels=output_train,
    path_record=join(FEATURES_PATH, "Train.tfrecords"),
)

print("[INFO]: Store Test TFRecord")
write_images_to_tfr(
    images=input_val,
    labels=output_val,
    path_record=join(FEATURES_PATH, "Test.tfrecords"),
)
