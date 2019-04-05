# import the necessary packages
import cv2


class SimplePreprocessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """
        Store the target image width, height, and interpolation method used
        when resizing
        :param width: target width of our input image after resizing
        :param height: target height of our input image after resizing
        :param inter: An optional parameter used to control which interpolation
            algorithm is used when resizing
        """

        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """
        Resize the image to a fixed size; ignoring the aspect ratio
        :param image: the image's numpy array
        :return: numpy array with shape (height, width)?
        """

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
