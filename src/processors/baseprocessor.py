# -*- coding: utf-8 -*-
"""
Set of function for loading image
"""
import tensorflow as tf

from src.config import DIMS_IMAGE


def load_image(image_path: str, preprocess_input=None) -> tf.Tensor:
    """
    Load an image
    Args:
        image_path (str): path of image
        preprocess_input(callable): preprocess function

    Returns:
        (tf.Tensor): image as tensor
    """
    # read the image from disk, decode it
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)

    # resize image with pad
    image = tf.image.resize_with_pad(
        image, target_height=DIMS_IMAGE[0], target_width=DIMS_IMAGE[0]
    )

    # preprocess image
    if preprocess_input:
        image = preprocess_input(image)

    # return the image
    return image
