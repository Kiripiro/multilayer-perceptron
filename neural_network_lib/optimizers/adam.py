import numpy as np

class Adam:
    """
    Adam optimizer implementation.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small value to prevent division by zero.
        m (list): List of first moment vectors.
        v (list): List of second moment vectors.
        t (int): Time step counter.

    Methods:
        __init__: Initializes the Adam optimizer with the specified hyperparameters.
        update: Updates the parameters using the computed gradients.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the Adam optimizer with the specified hyperparameters.

        Args:
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            beta1 (float): Exponential decay rate for the first moment estimates. Default is 0.9.
            beta2 (float): Exponential decay rate for the second moment estimates. Default is 0.999.
            epsilon (float): Small value to prevent division by zero. Default is 1e-8.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0

    def update(self, params, grads):
        """
        Updates the parameters using the computed gradients.

        Args:
            params (list): List of parameters to be updated.
            grads (list): List of gradients corresponding to the parameters.
        """
        if not self.m:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
