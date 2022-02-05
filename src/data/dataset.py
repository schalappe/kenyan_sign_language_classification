# -*- coding: utf-8 -*-
"""
Set of class for managing data
"""
from os.path import exists as check_exists_file
from typing import Generator

from h5py import File as H5File
from tensorflow.keras.utils import to_categorical


class DatasetWriter:
    """
    Store data into h5py dataset
    """

    def __init__(self, dims: tuple, output_path: str, buf_size: int = 1000) -> None:
        """
        Initialization
        Parameters
        ----------
        dims: tuple
            shape of dataset

        output_path: str
            path where to store the dataset

        buf_size: int
            length of the buffer
        """
        # check if the output path exists
        if check_exists_file(output_path):
            raise ValueError(
                "The output path already exists and cannot be "
                "overwritten. Manually delete it before continuing."
            )

        # store image/feature and class label
        self.database = H5File(output_path, "w")
        self.data = self.database.create_dataset("images", dims, dtype="float")
        self.labels = self.database.create_dataset("labels", (dims[0],), dtype="int")

        # buffer size and initialization
        self.buf_size = buf_size
        self.buffer = {"data": [], "labels": []}  # type: dict
        self.idx = 0

    def add(self, rows: list, labels: list) -> None:
        """
        Add data to the buffer
        Parameters
        ----------
        rows: list
            list of data

        labels: list
            list of labels
        """
        # add the rows and the labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.buf_size:
            self.flush()

    def flush(self) -> None:
        """
        Put data in dataset and empty th buffer
        """
        # write the buffer to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx : i] = self.buffer["data"]
        self.labels[self.idx : i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self) -> None:
        """
        Store the dataset into h5py file
        """
        # check if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.database.close()


class DatasetGenerator:
    """
    Generator for dataset
    """

    def __init__(self, db_path: str, binarize: bool, classes: int) -> None:
        """
        Initialization of generators
        Parameters
        ----------
        db_path: str
            Path of stored dataset

        binarize: bool
            True if it's needed to binarize label

        classes: int
            Number of classes
        """
        # store
        self.binarize = binarize if binarize else False
        self.classes = classes if binarize and classes else 1

        # open the HDF5 database
        self.database = H5File(db_path, "r")
        self.num_images = self.database["labels"].shape[0]

    def generator(self) -> Generator:
        """
        Continuously gives images and labels
        Returns
        -------
        Generator:
            Image && Label
        """
        for i in range(self.num_images):
            # extract the images and labels from the HDF dataset
            images = self.database["images"][i]
            labels = self.database["labels"][i]

            # check to see if the labels should be binarized
            if self.binarize:
                labels = to_categorical(labels, self.classes)

            # yield a tuple of images and labels
            yield images, labels

    def close(self) -> None:
        """
        Close the generator
        """
        # close the database
        self.database.close()
