import numpy as np
from neural_network_lib.models import Sequential
from .build_model import build_model
import os

def load_model(filename: str = 'data/model/model_weights_biases.npz') -> Sequential:
    """
    Load a pre-trained model's weights and biases from a file and assign them to a newly built model's layers.

    Args:
        filename (str): The name of the file containing the model's weights and biases (default is 'model_weights_biases.npz').

    Returns:
        Model: A model with the loaded weights and biases.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist. Make sure the model has been trained. ")

    data = np.load(filename, allow_pickle=True)
    weights = [data[key] for key in data.files[:len(data.files)//2]]
    biases = [data[key] for key in data.files[len(data.files)//2:]]

    model = build_model()
    
    weight_idx, bias_idx = 0, 0
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            layer.weights = weights[weight_idx]
            weight_idx += 1
        if hasattr(layer, 'biases'):
            layer.biases = biases[bias_idx]
            bias_idx += 1
    
    return model
