# -*- coding: utf-8 -*-
"""
    Set of  function used to process data
"""
from os.path import join
from typing import Any

import tensorflow as tf
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.config import CLASS_NAMES, DIMS_IMAGE, IMAGES_PATH, INPUT_CSV
from src.processors import load_image


def split_and_load_data(preprocess_input=None, test_size: float = 0.2) -> tuple:
    """
    Split dataset in train and test.
    Ratio: 80/20

    Returns
    -------
    tuple:
        List of data for training and testing
    """
    # extracts from csv file
    train_labels = read_csv(INPUT_CSV, header="infer")

    # set
    train_set = [join(IMAGES_PATH, image + ".jpg") for image in train_labels.img_IDS]
    label_set = list(train_labels.Label)

    # load data
    train_set = [
        load_image(path, preprocess_input)
        for path in tqdm(train_set, ncols=100, leave=False)
    ]

    # encode
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)
    label_set = label_encoder.transform(label_set)

    # split
    x_train, x_val, y_train, y_val = train_test_split(
        train_set, label_set, test_size=test_size, random_state=1337, stratify=label_set
    )

    return train_set, x_train, x_val, label_set, y_train, y_val


def _bytes_feature(value: Any) -> tf.train.Feature:
    """
    Returns a bytes_list from a string / byte.
    Parameters
    ----------
    value: Any
        Value to pack

    Returns
    -------
    tf.train.Feature:
        Packed value
    """
    if isinstance(value, tf.Tensor):
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
    """
    Returns a floast_list from a float / double.
    Parameters
    ----------
    value: float
        Value to pack

    Returns
    -------
    tf.train.Feature:
        Packed value
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value: Any) -> tf.train.Feature:
    """
    Returns an int64_list from a bool / enum / int / uint.
    Parameters
    ----------
    value: float
        Value to pack

    Returns
    -------
    tf.train.Feature:
        Packed value
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array: Any) -> Any:
    """
    Serialize an array (Generally raw image)
    Parameters
    ----------
    array: Any

    Returns
    -------
    Object:
        Serialized tensor
    """
    return tf.io.serialize_tensor(array)


def parse_single_image(image: tf.Tensor, label: int) -> tf.train.Example:
    """
    Parse a single image
    Parameters
    ----------
    image: tf.Tensor
        Image to pack

    label: int
        Label to pack

    Returns
    -------
    tf.train.Example:
        Sequence for train
    """
    data = {
        "image": _bytes_feature(serialize_array(image)),
        "label": _int64_feature(label),
    }

    return tf.train.Example(features=tf.train.Features(feature=data))


def parse_tfr_element(element: tf.train.Example) -> tuple:
    """
    Depack sequence of TFRecord
    Parameters
    ----------
    element: tf.train.Example
        Sequence of TFRecord

    Returns
    -------
    tuple:
        Image and Label
    """
    data = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    feature = tf.io.parse_tensor(content["image"], out_type=tf.float32)
    feature = tf.reshape(feature, shape=[DIMS_IMAGE[0], DIMS_IMAGE[1], DIMS_IMAGE[2]])

    return feature, content["label"]
