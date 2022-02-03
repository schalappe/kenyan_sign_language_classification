# -*- coding: utf-8 -*-
from numpy import ndarray
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    """
    Extract randomly a part of image
    """

    def __init__(self, width: int, height: int) -> None:
        """
        Initialization
        Args:
            width (int): width of new image
            height (int): height of new image
        """
        # width and height
        self.width = width
        self.height = height

    def preprocess(self, image: ndarray) -> ndarray:
        """
        Randomly extract a part of image
        Args:
            image (ndarray): Image to process

        Returns:
            (ndarray): new image
        """
        # extract a random crop from the image with the target width
        # and height
        assert isinstance(image, ndarray), "Image must be an numpy array"
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]
