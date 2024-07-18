import numpy as np

class Sequential:
    """
    Class representing a sequential neural network model.

    Attributes:
        layers (list): List of layers in the model.
        loss (object): Loss function used for optimization.
        optimizer (object): Optimizer used for updating model parameters.

    Methods:
        __init__: Initializes the Sequential model with empty layers, loss, and optimizer.
        add: Adds a layer to the model.
        compile: Configures the model with an optimizer and a loss function.
        forward: Performs forward pass through the model.
        backward: Performs backward pass through the model to compute gradients.
        fit: Trains the model on the given data for a specified number of epochs.
        evaluate: Evaluates the model on the given data.
    """
    
    def __init__(self):
        """
        Initializes the Sequential model with empty layers, loss, and optimizer.
        """
        self.layers = []
        self.loss = None
        self.optimizer = None
    
    def add(self, layer):
        """
        Adds a layer to the model.

        Args:
            layer (object): The layer to be added to the model.
        """
        self.layers.append(layer)
    
    def compile(self, optimizer, loss):
        """
        Configures the model with an optimizer and a loss function.

        Args:
            optimizer (object): The optimizer to use for training.
            loss (object): The loss function to use for training.
        """
        self.optimizer = optimizer
        self.loss = loss
    
    def forward(self, input):
        """
        Performs forward pass through the model.

        Args:
            input (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output of the model after the forward pass.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
        """
        Performs backward pass through the model to compute gradients.

        Args:
            gradient (numpy.ndarray): The gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input.
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def fit(self, X, y, epochs):
        """
        Trains the model on the given data for a specified number of epochs.

        Args:
            X (numpy.ndarray): The input data for training.
            y (numpy.ndarray): The target values for training.
            epochs (int): The number of epochs to train the model.

        Returns:
            dict: A dictionary containing the history of loss and accuracy for each epoch.
        """
        history = {'loss': [], 'accuracy': []}
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss_value = self.loss.forward(predictions, y)
            history['loss'].append(loss_value)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            history['accuracy'].append(accuracy)
            gradient = self.loss.backward(predictions, y)
            self.backward(gradient)
            params = []
            grads = []
            for layer in self.layers:
                if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                    params.extend([layer.weights, layer.biases])
                    grads.extend([layer.weights_gradient, layer.biases_gradient])
            self.optimizer.update(params, grads)
        return history
    
    def evaluate(self, X, y):
        """
        Evaluates the model on the given data.

        Args:
            X (numpy.ndarray): The input data for evaluation.
            y (numpy.ndarray): The target values for evaluation.

        Returns:
            tuple: A tuple containing the loss value and accuracy.
        """
        predictions = self.forward(X)
        loss_value = self.loss.forward(predictions, y)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return loss_value, accuracy
