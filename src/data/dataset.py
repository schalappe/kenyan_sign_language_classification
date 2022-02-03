# -*- coding: utf-8 -*-
"""
Set of class for managing data
"""
from os.path import exists as check_exists_file

from h5py import File as H5File
from numpy import arange, array
from tensorflow.keras.utils import to_categorical


class DatasetWriter:
    """
    Store data into h5py dataset
    """
    def __init__(self, dims: tuple, output_path: str, buf_size: int = 1000) -> None:
        """
        Initialization

        Args:
            dims (tuple): shape of dataset
            output_path (str): path where to store the dataset
            buf_size (int): length of the buffer
        """
        # check if the output path exists
        if check_exists_file(output_path):
            raise ValueError(
                "The output path already exists and cannot be "
                "overwritten. Manually delete it before continuing."
            )

        # store image/feature and class label
        self.db = H5File(output_path, "w")
        self.data = self.db.create_dataset("images", dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # buffer size and initialization
        self.buf_size = buf_size
        self.buffer = {"data": [], "labels": []}  # type: dict
        self.idx = 0

    def add(self, rows: list, labels: list) -> None:
        """
        Add data to the buffer

        Args:
            rows (list): list of data
            labels (list): list of labels
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
        self.db.close()


class DatasetGenerator:
    """
    Generator for dataset
    """

    def __init__(self, db_path: str, batch_size: int, binarize: bool, classes: int, preprocessors: list = None) -> None:
        """
        Initialization
        Args:
            db_path (str): Path of dataset stored
            batch_size (int): size of batch
            preprocessors (list): list of processors
        """
        # store
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.binarize = binarize if binarize else False
        self.classes = classes if binarize and classes else 1

        # open the HDF5 database
        self.db = H5File(db_path, "r")
        self.num_images = self.db["labels"].shape[0]

    def generator(self) -> tuple:
        """
        Continuously gives images and labels
        Returns:
            []: Array of images && Array of labels
        """
        for i in arange(0, self.num_images, self.batch_size):
            # extract the images and labels from the HDF dataset
            images = self.db["images"][i : i + self.batch_size]
            labels = self.db["labels"][i : i + self.batch_size]

            # check to see if the labels should be binarized
            if self.binarize:
                labels = to_categorical(labels, self.classes)

            # if our preprocessors are not None
            if self.preprocessors is not None:
                # the list of processed images
                proc_images = []

                # loop over the images
                for image in images:
                    # loop over the preprocessors
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    proc_images.append(image)
                images = array(proc_images)

            # yield a tuple of images and labels
            yield images, labels

    def close(self) -> None:
        """
        Close the generator
        """
        # close the database
        self.db.close()
