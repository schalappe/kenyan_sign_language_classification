# -*- coding: utf-8 -*-
from numpy import ndarray
from tensorflow.keras.utils import img_to_array


class ImageToArrayPreprocessor:
    """
    Convert Image in Array
    """

    def __init__(self, data_format: object = None) -> None:
        """
        Initialization
        Args:
            data_format (Any): format of image to convert
        """
        # store image
        self.dataFormat = data_format

    def preprocess(self, image: ndarray, normalize: bool = True) -> ndarray:
        """
        Convert Image to Array
        Args:
            image (ndarray): Image to convert
            normalize (bool): normalize or not

        Returns:
            (ndarray): Array for tensorflow model
        """
        if normalize:
            return img_to_array(image, data_format=self.dataFormat) / 255.0
        return img_to_array(image, data_format=self.dataFormat)
