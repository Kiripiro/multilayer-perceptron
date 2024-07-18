import numpy as np

class SGD:
    """
    A class representing the Stochastic Gradient Descent (SGD) optimizer.

    Parameters:
    - learning_rate (float): The learning rate to be used for updating the parameters. Default is 0.01.

    Methods:
    - __init__(self, learning_rate=0.01): Initializes the SGD optimizer with the specified learning rate.
    - update(self, params, grads): Updates the parameters based on the gradients using the SGD algorithm.
    """
    def __init__(self, learning_rate=0.01):
        """
        Initializes the SGD optimizer with the specified learning rate.

        Args:
            learning_rate (float): The learning rate to be used for updating the parameters. Default is 0.01.
        """
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        Updates the parameters based on the gradients using the SGD algorithm.

        Args:
            params (list): List of parameters to be updated.
            grads (list): List of gradients corresponding to the parameters.
        """
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

class SGDMomentum:
    """
    A class representing the Stochastic Gradient Descent with Momentum optimizer.

    Parameters:
    - learning_rate (float): The learning rate to be used for updating the parameters. Default is 0.01.
    - momentum (float): The momentum factor for the optimizer. Default is 0.9.

    Methods:
    - __init__(self, learning_rate=0.01, momentum=0.9): Initializes the SGDMomentum optimizer with the specified learning rate and momentum.
    - update(self, params, grads): Updates the parameters based on the gradients using the SGDMomentum algorithm.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initializes the SGDMomentum optimizer with the specified learning rate and momentum.

        Args:
            learning_rate (float): The learning rate to be used for updating the parameters. Default is 0.01.
            momentum (float): The momentum factor for the optimizer. Default is 0.9.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        """
        Updates the parameters based on the gradients using the SGDMomentum algorithm.

        Args:
            params (list): List of parameters to be updated.
            grads (list): List of gradients corresponding to the parameters.
        """
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            param += self.velocity[i]
