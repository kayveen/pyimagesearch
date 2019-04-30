from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.shallownet import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


print("[INFO] loading CIFAR-10 data...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

model = ShallowNet.build(width=32, height=32, depth=3, classes=10)

optimizer = SGD(lr=0.005)
model.compile(metrics=["accuracy"], loss="categorical_crossentropy",
              optimizer=optimizer)

print ("[INFO] training network...")

H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=40,
              batch_size=32, verbose=1)

print ("[INFO] evaluating network...")

predictions = model.predict(x=testX, batch_size=32)

print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=["airplane", "automobile", "bird", "cat", "deer", "dog",
                  "frog", "horse", "ship", "truck"]
))


# plot the results
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
