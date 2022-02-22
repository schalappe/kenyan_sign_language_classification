# -*- coding: utf-8 -*-
"""
Set of function use to augment image
"""
import tensorflow as tf


@tf.function
def random_hue(images: tf.Tensor) -> tf.Tensor:
    """
    Adjust the hue of RGB images by a random factor
    Parameters
    ----------
    images: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    if len(images.shape) == 3:
        return tf.image.random_hue(images, 0.08)

    return tf.map_fn(fn=lambda image: tf.image.random_hue(image, 0.08), elems=images)


@tf.function
def random_saturation(images: tf.Tensor) -> tf.Tensor:
    """
    Adjust the saturation of RGB images by a random factor
    Parameters
    ----------
    images: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    if len(images.shape) == 3:
        return tf.image.random_saturation(images, 0.6, 1.6)

    return tf.map_fn(
        fn=lambda image: tf.image.random_saturation(image, 0.6, 1.6), elems=images
    )


@tf.function
def random_brightness(images: tf.Tensor) -> tf.Tensor:
    """
    Adjust the brightness of images by a random factor
    Parameters
    ----------
    images: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    if len(images.shape) == 3:
        return tf.image.random_brightness(images, 0.05)

    return tf.map_fn(
        fn=lambda image: tf.image.random_brightness(image, 0.05), elems=images
    )


@tf.function
def random_contrast(images: tf.Tensor) -> tf.Tensor:
    """
    Adjust the contrast of an image or images by a random factor
    Parameters
    ----------
    images: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Augmented image
    """
    if len(images.shape) == 3:
        return tf.image.random_contrast(images, 0.7, 1.3)

    return tf.map_fn(
        fn=lambda image: tf.image.random_contrast(image, 0.7, 1.3), elems=images
    )


@tf.function
def gaussian_noise(images: tf.Tensor) -> tf.Tensor:
    """
    Add gaussian noise to an image
    Parameters
    ----------
    images: tf.Tensor
        Image to process

    Returns
    -------
    tf.Tensor
        Noised image
    """

    def noised_image(image: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(
            shape=tf.shape(image), mean=0.1, stddev=0.1, dtype=image.dtype
        )
        noise_img = tf.add(image, noise)
        return tf.clip_by_value(noise_img, 0.0, 1.0)

    if len(images.shape) == 3:
        return noised_image(images)

    return tf.map_fn(fn=lambda image: noised_image(image), elems=images)


@tf.function
def random_augmentation(
    images: tf.Tensor, percentage: float = 0.75, noise: bool = False
) -> tf.Tensor:
    """
    Apply random augmentation
    Parameters
    ----------
    images: tf.Tensor
        Image to augment

    percentage: float
        Percentage of case where apply augmentation

    noise: bool
        Add noise to augmentation

    Returns
    -------
    tf.Tensor:
        Augmented images
    """
    augmentation = [
        random_hue,
        random_saturation,
        random_brightness,
        random_contrast,
    ]

    # add noise or not
    if noise:
        augmentation.append(gaussian_noise)

    # apply augmentation
    for func in augmentation:
        # Apply an augmentation
        if tf.random.uniform([], 0, 1) > percentage:
            images = func(images)

    return images
