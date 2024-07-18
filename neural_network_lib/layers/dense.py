import numpy as np
from neural_network_lib.initializers import *

class Dense:
    """
    Class implementing a dense layer for a neural network.

    Attributes:
        weights (ndarray): The weights of the dense layer.
        biases (ndarray): The biases of the dense layer.

    Methods:
        __init__(self, input_size, output_size): Initializes the dense layer with random weights and zeros for biases.
        forward(self, input): Performs the forward pass of the dense layer.
        backward(self, gradient): Performs the backward pass of the dense layer to calculate gradients.
    """
    def __init__(self, input_size, output_size, kernel_initializer=None):
        """
        Initializes the dense layer with random weights and zeros for biases.

        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output features.
        """
        shape = (input_size, output_size)
        if kernel_initializer == 'xavier':
            self.weights = xavier_initialization(shape)
        elif kernel_initializer == 'he':
            self.weights = he_initialization(shape)
        elif kernel_initializer == 'lecun':
            self.weights = lecun_initialization(shape)
        else:
            self.weights = random_initialization(shape)

        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        """
        Performs the forward pass of the dense layer.

        Args:
            input (ndarray): The input data.

        Returns:
            ndarray: The output of the dense layer.
        """
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, gradient):
        """
        Performs the backward pass of the dense layer to calculate gradients.

        Args:
            gradient (ndarray): The gradient of the loss with respect to the output.

        Returns:
            ndarray: The gradient of the loss with respect to the input.
        """
        self.weights_gradient = np.dot(self.input.T, gradient)
        self.biases_gradient = np.mean(gradient, axis=0, keepdims=True)
        input_gradient = np.dot(gradient, self.weights.T)
        return input_gradient
