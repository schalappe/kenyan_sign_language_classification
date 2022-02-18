# -*- coding: utf-8 -*-
"""
    Script used to create a TFRecordDataset that contains the data for training
"""
from os.path import join

from addons import model_preprocess
from src.config import FEATURES_PATH
from src.data import split_and_load_data, write_images_to_tfr

for index, model in enumerate(model_preprocess.keys()):
    print(f"f[INFO]: Model {index} /{len(model_preprocess)}")

    print("[INFO]: Split and Load dataset")
    (
        inputs,
        input_train,
        input_val,
        outputs,
        output_train,
        output_val,
    ) = split_and_load_data(model_preprocess[model])

    print("[INFO]: Store All TFRecord")
    write_images_to_tfr(
        images=inputs,
        labels=outputs,
        path_record=join(FEATURES_PATH, f"All_{model}.tfrecords"),
    )

    print("[INFO]: Store Train TFRecord")
    write_images_to_tfr(
        images=input_train,
        labels=output_train,
        path_record=join(FEATURES_PATH, f"Train_{model}.tfrecords"),
    )

    print("[INFO]: Store Test TFRecord")
    write_images_to_tfr(
        images=input_val,
        labels=output_val,
        path_record=join(FEATURES_PATH, f"Test_{model}.tfrecords"),
    )
