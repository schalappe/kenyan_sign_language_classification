# -*- coding: utf-8 -*-
"""
Set of function for pipeline
"""
from os.path import exists as check_exist_file

import tensorflow as tf

from src.config import CLASS_NAMES
from src.processors import extract_patches, random_augmentation

from .utils import parse_tfr_element


def prepare_from_tfrecord(tfrecord: str, batch: int, train: bool, dims: tuple):
    """
    Prepare dataset for Training and Testing model
    Parameters
    ----------
    tfrecord: str
        Path of TFRecord

    batch: int
        Number of batch

    train: bool
        Add augmentation to data or not

    dims: tuple
        New size of image

    Returns
    -------
    tf.data.Dataset
    """
    assert check_exist_file(tfrecord), "Le TFRecord n'existe pas"

    # load data for TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord)

    # decode and resize image
    dataset = dataset.map(parse_tfr_element).cache()
    if train:
        dataset = dataset.map(
            lambda image, label: (
                extract_patches(image, dims[0], dims[1]),
                tf.one_hot(label, len(CLASS_NAMES)),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda image, label: (
                tf.image.resize(image, [dims[0], dims[1]]),
                tf.one_hot(label, len(CLASS_NAMES)),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    dataset = dataset.cache()

    # shuffle if it's training
    if train:
        dataset = dataset.shuffle(batch * 100).cache()

    # infini dataset
    dataset = dataset.repeat()

    # Batch all datasets
    dataset = dataset.batch(batch)

    # Use data augmentation only on the training set.
    if train:
        # Apply an augmentation only in 50% of the cases.
        dataset = dataset.map(
            lambda image, label: (
                random_augmentation(image, percentage=0.5, noise=True),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Use buffered prefetching on all datasets
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
