from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense
from tensorflow.keras import backend as K


class LeNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(
            20,
            kernel_size=(5, 5),
            padding="same",
            input_shape=inputShape
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(strides=(2, 2)))
        model.add(Conv2D(
            50,
            kernel_size=(5, 5),
            padding="same"
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model