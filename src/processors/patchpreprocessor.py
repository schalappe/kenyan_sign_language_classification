# -*- coding: utf-8 -*-
"""
Set of function for cropping an image
"""
from random import uniform

import tensorflow as tf


@tf.function
def extract_patches(images: tf.Tensor, height: int, width: int) -> tf.Tensor:
    """
    Randomly extract a part of image

    Parameters
    ----------
    images: tf.Tensor
        Image to process

    height: int
        Height of the crop

    width: int
        Width of the crop

    Returns
    -------
    patch: tf.Tensor
        New image
    """
    assert isinstance(images, tf.Tensor), "Image must be a Tensor"

    def extract_randomly(image) -> tf.Tensor:
        # extract many crop from the image with the target width
        # and height
        patches = tf.image.extract_patches(
            tf.reshape(image, shape=(-1, *image.shape)),
            [1, height, width, 1],
            [1, 8, 8, 1],
            [1, 1, 1, 1],
            padding="VALID",
        )
        patches_res = tf.reshape(patches, shape=(-1, height, width, image.shape[2]))

        # randomly choose a crop
        index = int(uniform(0, patches_res.shape[0] - 1))

        return patches_res[index]

    if len(images.shape) == 3:
        return extract_randomly(images)

    return tf.map_fn(fn=lambda image: extract_randomly(image), elems=images)
