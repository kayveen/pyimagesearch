# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse


# load the training and testing data, then scale it into the range [0,1]
print('[INFO] Loading CIFAR-10 data....')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the optimize and model
print('[INFO] compiling model...')
opt = SGD(lr=0.01, decay=0.01/16, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# construct the callback to save only the best model to disk
# based on the validation loss
checkpoint = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=32,
                         write_graph=True, write_grads=False,
                         write_images=False)
callbacks = [checkpoint]

# train the network
print('[INFO] training network....')
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=16, callbacks=callbacks, verbose=1)