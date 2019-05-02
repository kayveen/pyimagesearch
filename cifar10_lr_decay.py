import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

def step_decay(epoch):

    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1+epoch) / dropEvery))

    return float(alpha)


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to plot")
args = vars(ap.parse_args())

print("[INFO] loading cifar10 data")

(trainX, trainY), (testX, testY)= cifar10.loadData()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]

# define the set of callbacks to be passed to the model during
# training
callbacks = [LearningRateScheduler(step_decay)]
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64,
              epochs=40, callbacks=callbacks, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])