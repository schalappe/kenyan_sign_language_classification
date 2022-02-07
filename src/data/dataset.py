# -*- coding: utf-8 -*-
"""
Set of class for managing data
"""
import tensorflow as tf
from tqdm import tqdm

from src.data.utils import parse_single_image


def write_images_to_tfr(images: tuple, labels: tuple, path_record: str) -> None:
    """
    Create a writer to store data
    Parameters
    ----------
    images: tuple
        List of images

    labels: tuple
        List of labels

    path_record: str
        Path where store TFRecord
    """
    count = 0
    with tf.io.TFRecordWriter(path_record) as file_writer:
        for index in tqdm(range(len(images)), ncols=100):
            # get the data we want to write
            current_image = images[index]
            current_label = labels[index]

            # add sequence to writer
            file_writer.write(
                parse_single_image(
                    image=current_image, label=current_label
                ).SerializeToString()
            )
            count += 1

    print(f"Wrote {count} elements to TFRecord")
