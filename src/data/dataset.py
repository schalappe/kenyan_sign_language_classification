# -*- coding: utf-8 -*-
"""
Set of class for managing data
"""
from os.path import join

import tensorflow as tf
from tqdm import tqdm

from src.config import DIMS_MODEL, FEATURES_PATH
from src.data.utils import parse_single_image

from .pipeline import prepare_from_tfrecord


def write_images_to_tfr(images: tuple, labels: tuple, path_record: str) -> None:
    """
    Create a writer to store data
    Parameters
    ----------
    images: tuple
        List of images

    labels: tuple
        List of labels

    path_record: str
        Path where store TFRecord
    """
    count = 0
    with tf.io.TFRecordWriter(path_record) as file_writer:
        for index in tqdm(range(len(images)), ncols=100, leave=False):
            # get the data we want to write
            current_image = images[index]
            current_label = labels[index]

            # add sequence to writer
            file_writer.write(
                parse_single_image(
                    image=current_image, label=current_label
                ).SerializeToString()
            )
            count += 1

    print(f"Wrote {count} elements to TFRecord")


def return_dataset(family_model: str, batch_size: int) -> tuple:
    train_set = prepare_from_tfrecord(
        tfrecord=join(FEATURES_PATH, f"Train_{family_model}.tfrecords"),
        batch=batch_size,
        train=True,
        dims=DIMS_MODEL,
    )

    test_set = prepare_from_tfrecord(
        tfrecord=join(FEATURES_PATH, f"Test_{family_model}.tfrecords"),
        batch=batch_size,
        train=False,
        dims=DIMS_MODEL,
    )

    return train_set, test_set
