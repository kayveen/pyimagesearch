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
