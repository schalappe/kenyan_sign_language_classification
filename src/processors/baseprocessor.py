# -*- coding: utf-8 -*-
"""
Set of function for loading image
"""
from os.path import exists as check_exist_file

import tensorflow as tf


def load_image(image_path: str) -> tf.Tensor:
    """
    Load an image
    Args:
        image_path (str): path of image

    Returns:
        (tf.Tensor): image as tensor
    """
    # check if path is goog
    assert check_exist_file(image_path), "Image doesn't exist."

    # read the image from disk, decode it
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)

    # return the image
    return image
