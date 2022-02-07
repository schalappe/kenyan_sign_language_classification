# -*- coding: utf-8 -*-
"""
Set of function for pipeline
"""
from os.path import exists as check_exist_file

import tensorflow as tf

from src.config import CLASS_NAMES, DIMS_MODEL
from src.processors import random_augmentation

from .utils import parse_tfr_element


def prepare_from_tfrecord(tfrecord: str, batch: int, train: bool):
    assert check_exist_file(tfrecord), "Le TFRecord n'existe pas"

    # load data for TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord)

    # decode and resize image
    dataset = dataset.map(parse_tfr_element).cache()
    dataset = dataset.map(
        lambda image, label: (
            tf.image.resize(image, [DIMS_MODEL[0], DIMS_MODEL[0]]),
            tf.one_hot(label, len(CLASS_NAMES)),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # shuffle if it's training
    dataset = dataset.shuffle(1000).cache()

    # Batch all datasets
    dataset = dataset.batch(batch)

    # Use data augmentation only on the training set.
    if train:
        # Apply an augmentation only in 75% of the cases.
        dataset = dataset.map(
            lambda image, label: (random_augmentation(image), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Use buffered prefetching on all datasets
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
