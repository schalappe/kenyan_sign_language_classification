# -*- coding: utf-8 -*-
"""
Set of function for resizing an image
"""
import tensorflow as tf


def simple_resize(
    image: tf.Tensor, height: int, width: int, method: str = "bilinear"
) -> tf.Tensor:
    """
    Resize an image
    Args:
        image (tf.Tensor): image to resize
        height (int): new height
        width (int): new width
        method (str): resize method

    Returns:
        (tf.Tensor): new resized image
    """
    assert isinstance(image, tf.Tensor), "Image must be a Tensor"
    return tf.image.resize(image, (height, width), method=method) / 225.0


def aspect_resize(
    image: tf.Tensor, height: int, width: int, method: str = "bilinear"
) -> tf.Tensor:
    """
    Resize an image, keep aspect
    Args:
        image (tf.Tensor): image to resize
        height (int): new height
        width (int): new width
        method (str): resize method

    Returns:
        (tf.Tensor): new resized image
    """
    assert isinstance(image, tf.Tensor), "Image must be a Tensor"
    # grab the dimensions of the image and then initialize
    # the deltas to use when cropping
    (i_height, i_width) = image.shape[:2]
    d_height, d_width = 0, 0

    # crop
    if i_width < i_height:
        # calculate the ratio of the width and construct the
        # dimensions
        ratio = width / float(i_width)
        dim = (int(i_height * ratio), width)
        resized = tf.image.resize(image, dim, method=method)
        d_height = int((resized.shape[0] - height) / 2.0)
    else:
        # calculate the ratio of the height and construct the
        # dimensions
        ratio = height / float(i_height)
        dim = (height, int(i_width * ratio))
        resized = tf.image.resize(image, dim, method=method)
        d_width = int((resized.shape[1] - width) / 2.0)

    # resize the image
    (i_height, i_width) = resized.shape[:2]
    resized = resized[d_height : i_height - d_height, d_width : i_width - d_width]

    return tf.image.resize(resized, (height, width), method=method) / 255.0
