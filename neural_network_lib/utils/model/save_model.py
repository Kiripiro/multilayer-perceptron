import numpy as np
from typing import Any
import os

def save_model(model: Any, filename: str = 'model_weights_biases.npz') -> None:
    """
    Save the weights and biases of a model's layers to a specified file in .npz format using NumPy.

    Args:
        model (Any): The model object containing layers with weights and biases.
        filename (str): The name of the file to save the weights and biases to (default is 'model_weights_biases.npz').
    """
    directory = 'data/model'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)

    weights = [layer.weights for layer in model.layers if hasattr(layer, 'weights')]
    biases = [layer.biases for layer in model.layers if hasattr(layer, 'biases')]
    np.savez(filepath, *weights, *biases)