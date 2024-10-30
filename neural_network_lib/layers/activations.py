import numpy as np

class ReLU:
    """
    ReLU activation function.
    
    Methods:
        forward(input): Applies the ReLU activation function to the input.
        backward(gradient): Computes the gradient of the ReLU activation function.
    """
    def forward(self, input):
        """
        Applies the ReLU activation function to the input.
        
        Args:
            input (numpy.ndarray): The input data.
        
        Returns:
            numpy.ndarray: The output of the ReLU activation function.
        """
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, gradient):
        """
        Computes the gradient of the ReLU activation function.
        
        Args:
            gradient (numpy.ndarray): The gradient of the loss with respect to the output.
        
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input.
        """
        return gradient * (self.input > 0)

class Sigmoid:
    """
    Sigmoid activation function.
    
    Methods:
        forward(input): Applies the Sigmoid activation function to the input.
        backward(gradient): Computes the gradient of the Sigmoid activation function.
    """
    def forward(self, input):
        """
        Applies the Sigmoid activation function to the input.
        
        Args:
            input (numpy.ndarray): The input data.
        
        Returns:
            numpy.ndarray: The output of the Sigmoid activation function.
        """
        self.output = 1 / (1 + np.exp(-np.clip(input, -500, 500)))
        return self.output
    
    def backward(self, gradient):
        """
        Computes the gradient of the Sigmoid activation function.
        
        Args:
            gradient (numpy.ndarray): The gradient of the loss with respect to the output.
        
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input.
        """
        return gradient * self.output * (1 - self.output)

class Softmax:
    """
    Softmax activation function.
    
    Methods:
        forward(input): Applies the Softmax activation function to the input.
        backward(gradient): Computes the gradient of the Softmax activation function.
    """
    def forward(self, input):
        """
        Applies the Softmax activation function to the input.
        
        Args:
            input (numpy.ndarray): The input data.
        
        Returns:
            numpy.ndarray: The output of the Softmax activation function.
        """
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, gradient):
        """
        Computes the gradient of the Softmax activation function.
        
        Args:
            gradient (numpy.ndarray): The gradient of the loss with respect to the output.
        
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input.
        """
        return gradient
