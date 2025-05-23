import numpy as np
from neural_network_lib.metrics.metrics import precision_recall_f1_score

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
        predict: Generates class predictions or probabilities from input data.
        get_last_layer: Returns the last layer of the model.
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
            preds = self.forward(X)
            loss_value = self.loss.forward(preds, y)
            history['loss'].append(loss_value)

            if preds.ndim > 1 and preds.shape[1] > 1:
                y_true_flat = np.argmax(y, axis=1)
                y_pred_flat = np.argmax(preds, axis=1)
            else:
                y_true_flat = np.array(y).ravel()
                y_pred_flat = (preds >= 0.5).astype(int).ravel()

            accuracy = np.mean(y_pred_flat == y_true_flat)
            history['accuracy'].append(accuracy)

            grad = self.loss.backward(preds, y)
            self.backward(grad)
            params, grads = [], []
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
        preds = self.forward(X)
        loss  = self.loss.forward(preds, y)

        precision, recall, f1, labels = precision_recall_f1_score(y, preds)

        if preds.ndim > 1 and preds.shape[1] > 1:
            y_true_flat = np.argmax(y, axis=1)
            y_pred_flat = np.argmax(preds, axis=1)
        else:
            y_true_flat = np.array(y).ravel()
            y_pred_flat = (preds >= 0.5).astype(int).ravel()
        accuracy = np.mean(y_pred_flat == y_true_flat)

        print(f'Loss: {loss:.4f}  Acc: {accuracy:.4f}')
        for lab, p, r, f in zip(labels, precision, recall, f1):
            print(f'Class {lab}: P={p:.2f} R={r:.2f} F1={f:.2f}')
        return loss, accuracy

    def predict(self, X, return_probs=False):
        proba = self.forward(X)
        if return_probs:
            return proba
        if proba.ndim > 1 and proba.shape[1] > 1:
            return np.argmax(proba, axis=1)
        return (proba >= 0.5).astype(int).ravel()

    def get_last_layer(self):
        return self.layers[-1] if self.layers else None
