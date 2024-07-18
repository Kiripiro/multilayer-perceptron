import numpy as np

def random_initialization(shape):
    """
    Randomly initializes the weights of a layer with a given shape.
    
    Args:
        shape (tuple): The shape of the weight matrix.
        
    Returns:
        np.ndarray: A weight matrix of the given shape, randomly initialized with small values.
    """
    return np.random.randn(*shape) * 0.01

def xavier_initialization(shape):
    """
    Initializes the weights of a layer using Xavier initialization.
    
    Args:
        shape (tuple): The shape of the weight matrix.
        
    Returns:
        np.ndarray: A weight matrix of the given shape, initialized using the Xavier method.
    """
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def he_initialization(shape):
    """
    Initializes the weights of a layer using He initialization.
    
    Args:
        shape (tuple): The shape of the weight matrix.
        
    Returns:
        np.ndarray: A weight matrix of the given shape, initialized using the He method.
    """
    fan_in = shape[0]
    std_dev = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std_dev

def lecun_initialization(shape):
    """
    Initializes the weights of a layer using LeCun initialization.
    
    Args:
        shape (tuple): The shape of the weight matrix.
        
    Returns:
        np.ndarray: A weight matrix of the given shape, initialized using the LeCun method.
    """
    fan_in = shape[0]
    std_dev = np.sqrt(1.0 / fan_in)
    return np.random.randn(*shape) * std_dev
