class InputLayer:
    """
    Class representing an input layer in a neural network.

    Attributes:
        input_size (int): The size of the input data.

    Methods:
        __init__: Initializes the InputLayer with the specified input size.
        forward: returns x
        backward: returns gradients

    No operations should be done since it's just representing x features
    """
    def __init__(self, input_size):
        self.input_size = input_size

    def forward(self, x):
        """
        Forward pass for the input layer. Simply returns the input as is.
        
        Args:
            x (numpy.ndarray): The input data.
        
        Returns:
            numpy.ndarray: The input data, unchanged.
        """
        return x

    def backward(self, gradient):
        """
        Backward pass for the input layer. Simply returns the gradient as is.
        
        Args:
            gradient (numpy.ndarray): The gradient from the subsequent layer.
        
        Returns:
            numpy.ndarray: The gradient, unchanged.
        """
        return gradient
