# import the necessary packages
import numpy as np
import cv2
import os


class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):

        # Store the image preprocessor
        self.preprocessors = preprocessors

        # If preprocessors are None, initialize them to an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=1):
        """
        Load images and return tuple of data and labels in numpy array
        :param imagePaths: a list specifying the file paths to the images in our dataset residing on disk
        :param verbose: print updates to the console to monitor how many images processed, each verbose image
        :return: a tuple of data and labels
        """

        # Initialise a list of features and labels

        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the label class assuming that has the following format
            # /path/to/dataset/{class}/{image}.jpg

            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image

                for p in self.preprocessors:
                    p.preprocess(image)

            #  treat our processed image as a "feature vector"
            # by updating the data list followed by the labels

            data.append(image)
            labels.append(label)

            # Show an update every 'verbose' image
            if verbose > 0 and i > 0 and (i+1) % verbose  == 0:
                print(f'[INFO] processed {i+1}/{len(imagePaths)}')

        # return a tuple of the data and labels
        return np.array(data), np.array(labels)
