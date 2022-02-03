# -*- coding: utf-8 -*-
from cv2 import INTER_AREA, resize
from imutils import resize as resize_im
from numpy import ndarray


class SizePreprocessor:
    """
    Class used to resize image
    """

    def __init__(self, width: int, height: int, inter: int = INTER_AREA) -> None:
        """
        Initialization

        Args:
            width (int): new width of image
            height (int): new height of image
            inter (int): interpolation for resizing
        """
        # store the target image width, height, and interpolation
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: ndarray) -> None:
        pass


class SimplePreprocessor(SizePreprocessor):
    """
    Class used to resize image
    """

    def __init__(self, width: int, height: int, inter: int = INTER_AREA) -> None:
        """
        Initialization

        Args:
            width (int): new width of image
            height (int): new height of image
            inter (int): interpolation for resizing
        """
        super().__init__(width, height, inter)

    def preprocess(self, image: ndarray) -> None:
        """
        Resize an image
        Args:
            image (ndarray): image to resize

        Returns:
            (ndarray): new image
        """
        assert isinstance(image, ndarray), "Image must be an numpy array"
        return resize(image, (self.width, self.height), interpolation=self.inter)


class AspectAwarePreprocessor(SizePreprocessor):
    """
    Class used to resize image while keeping the ratio
    """

    def __init__(self, width: int, height: int, inter: int = INTER_AREA) -> None:
        """
        Initialization

        Args:
            width (int): new width of image
            height (int): new height of image
            inter (int): interpolation for resizing
        """
        super().__init__(width, height, inter)

    def preprocess(self, image: ndarray) -> ndarray:
        """
        Resize an image
        Args:
            image (ndarray): image to resize

        Returns:
            (ndarray): new image
        """
        assert isinstance(image, ndarray), "Image must be an numpy array"
        # grab the dimensions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW, dH = 0, 0

        # crop
        if w < h:
            image = resize_im(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = resize_im(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]

        return resize(image, (self.width, self.height), interpolation=self.inter)