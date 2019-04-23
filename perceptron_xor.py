from nn.perceptron import Perceptron
import numpy as np

# construct the OR dataset

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our perceptron and train it
print("[INFO] training perceptron...")

perceptron = Perceptron(X.shape[1], alpha=0.1)
perceptron.fit(X, y, epochs=20)

# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):

    prediction = perceptron.predict(x)
    print(f"[INFO] data={x}, ground-truth={target[0]}, pred={prediction}")