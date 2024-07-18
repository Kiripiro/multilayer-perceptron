import yaml
import importlib
from neural_network_lib.models import Sequential
import os
import sys

def validate_config(config):
    """
    Validate the configuration dictionary to ensure it contains the required keys.

    Args:
    config (dict): The configuration dictionary to be validated.

    Raises:
    ValueError: If the configuration dictionary is missing any of the required keys.

    Returns:
    None
    """
    required_keys = {'layers'}
    if not all(key in config for key in required_keys):
        raise ValueError("Invalid configuration file: missing 'layers' key")
    for layer in config['layers']:
        if 'type' not in layer:
            raise ValueError(f"Invalid configuration file: layer missing 'type' key: {layer}")
        if layer['type'] == 'Dense':
            if 'input_size' not in layer or 'output_size' not in layer:
                raise ValueError(f"Invalid configuration file: Dense layer missing 'input_size' or 'output_size': {layer}")

def build_model() -> Sequential:
    """
    Builds the model from the YAML file located in the /config folder.

    Args:
    config_path (str): Optional path to the configuration file.

    Returns:
    Sequential: A neural network model constructed based on the YAML configuration.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "..", "config", "model_config.yaml")
    
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            validate_config(config)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None
        except ValueError as exc:
            print(f"{exc}")
            sys.exit(1)

    model = Sequential()

    for layer_config in config['layers']:
        layer_type = layer_config.pop('type')
        try:
            module = importlib.import_module('neural_network_lib.layers')
            LayerClass = getattr(module, layer_type)
        except (ImportError, AttributeError) as exc:
            print(f"Error importing layer {layer_type}: {exc}")
            return None
        try:
            model.add(LayerClass(**layer_config))
        except TypeError as exc:
            print(f"Error initializing layer {layer_type} with config {layer_config}: {exc}")
            return None

    return model