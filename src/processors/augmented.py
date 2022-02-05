# -*- coding: utf-8 -*-
"""
Set of function use to augment image
"""
import tensorflow as tf


def random_hue(image: tf.Tensor) -> tf.Tensor:
    """
    Adjust the hue of RGB images by a random factor
    Parameters
    ----------
    image: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    return tf.image.random_hue(image, 0.08)


def random_saturation(image: tf.Tensor) -> tf.Tensor:
    """
    Adjust the saturation of RGB images by a random factor
    Parameters
    ----------
    image: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    return tf.image.random_saturation(image, 0.6, 1.6)


def random_brightness(image: tf.Tensor) -> tf.Tensor:
    """
    Adjust the brightness of images by a random factor
    Parameters
    ----------
    image: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    return tf.image.random_brightness(image, 0.05)


def random_contrast(image: tf.Tensor) -> tf.Tensor:
    """
    Adjust the contrast of an image or images by a random factor
    Parameters
    ----------
    image: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    return tf.image.random_contrast(image, 0.7, 1.3)


def gaussian_noise(image: tf.Tensor) -> tf.Tensor:
    """
    Add gaussian noise to an image
    Parameters
    ----------
    image: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Noised image
    """
    noise = tf.random.normal(
        shape=tf.shape(image), mean=0.1, stddev=0.1, dtype=image.dtype
    )
    noise_img = tf.add(image, noise)
    return tf.clip_by_value(noise_img, 0.0, 1.0)
