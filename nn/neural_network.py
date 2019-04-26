import numpy as np


class NeuralNetwork:

    def __init__(self, layers, alpha=0.1):
        """

        :param layers: A list of integers which represents the actual architecture of the feedforward
                network. For example, a value of [2;2;1] would imply that our first input layer has two nodes,
                our hidden layer has two nodes, and our final output layer has one node
        :param alpha: learning rate of our neural network
        """
        # initialize the list of weights matrices, then store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but
        # stop before we reach the last two layers

        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the
            # number of nodes in each respective layer together
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

            # the last two layers are a special case where the input
            # connections need a bias term but the output does not

            w = np.random.randn(layers[-2] + 1, layers[-1])
            self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return f'NeuralNetwork : {"-".join(str(l) for l in self.layers)}'

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function ASSUMING
        #  that ‘x‘ has already been passed through the ‘sigmoid‘ function
        return x * (1 - x)


    def fit(self, X, y, epochs=100, displayUpdate=10):
        # Bias trick
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):

            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))


    def fit_partial(self, x, y):

        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(len(self.W)):
            # taking the dot product between the activation and
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = np.dot(A[layer], self.W[layer])
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of
            # activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activations list) and the true target
        # value

        error = A[-1] - y

        # from here, we need to apply the chain rule and build our
        # list of deltas ‘D‘; the first entry in the deltas is
        # simply the error of the output layer times the derivative
        # of our activation function for the output value

        D = [error * self.sigmoid_deriv(A[-1])]

        # once you understand the chain rule it becomes super easy
        # to implement with a ‘for‘ loop -- simply loop over the
        # layers in reverse order (ignoring the last two since we
        # already have taken them into account)

        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer

            delta = np.dot(D[-1], self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we looped over our layers in reverse order we need to
        # reverse the deltas

        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual "learning" takes
            # place
            self.W[layer] = self.W[layer] - self.alpha * np.dot(A[layer].T, D[layer])

    def predict(self, X, addBias=True):

        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):

            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):

        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = np.sum((predictions - targets) ** 2) / 2

        return loss
