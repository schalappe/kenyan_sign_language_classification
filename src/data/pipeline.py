# -*- coding: utf-8 -*-
"""
Set of function for pipeline
"""
import tensorflow as tf

from src.config import CLASS_NAMES, DIMS_IMAGE
from src.data import DatasetGenerator
from src.processors import (
    extract_patches,
    gaussian_noise,
    random_brightness,
    random_contrast,
    random_hue,
    random_saturation,
    rescale_image,
    simple_resize,
)


def prepare(
    dataset: tf.data.Dataset,
    batch: int,
    height: int,
    width: int,
    shuffle: bool,
    augment: bool,
) -> tf.data.Dataset:
    """
    Apply preprocessing function to a dataset
    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to prepare

    batch: int
        Batch size

    height: int
        Height of resized image

    width: int
        Width of resized image

    shuffle: bool
        True if it's needed to shuffle data

    augment: bool
        True if it's needed to augment data

    Returns
    -------
    tf.data.Dataset:
        Prepared dataset
    """
    # Resize image
    if augment:
        dataset = dataset.map(
            lambda image, label: (extract_patches(image, height, width), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda image, label: (simple_resize(image, height, width), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Rescale image
    dataset = dataset.map(
        lambda image, label: (rescale_image(image), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(1000)

    # Batch all datasets
    dataset = dataset.batch(batch)

    # Use data augmentation only on the training set.
    if augment:
        augmentation = [
            random_hue,
            random_saturation,
            random_brightness,
            random_contrast,
            gaussian_noise,
        ]
        for func in augmentation:
            dataset = dataset.cache()
            # Apply an augmentation only in 50% of the cases.
            dataset.map(
                lambda image, label: (
                    tf.cond(
                        tf.random.uniform([], 0, 1) > 0.75,
                        lambda: func(image),
                        lambda: image,
                    ),
                    label,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    # Use buffered prefetching on all datasets
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def load_dataset_from_generator(
    generator: DatasetGenerator, batch: int, shape: int, train: bool
) -> tf.data.Dataset:
    """
    Load a dataset from generator
    Parameters
    ----------
    generator: DatasetGenerator
        Generator of image

    batch: int
        Batch size

    shape:
        Heigth and Width (same)

    train: bool
        If True, shuffle and augment data

    Returns
    -------
    tf.data.Dataset:
        Dataset ready to train or test
    """
    # output shapes and types
    output_types = (tf.float64, tf.uint8)
    output_shapes = (DIMS_IMAGE, (len(CLASS_NAMES),))

    # load from generator
    dataset = tf.data.Dataset.from_generator(
        generator.generator, output_shapes=output_shapes, output_types=output_types
    )

    # prepare dataset
    dataset = prepare(
        dataset=dataset,
        batch=batch,
        height=shape,
        width=shape,
        shuffle=train,
        augment=train,
    )

    return dataset
