import numpy as np
import yaml
import os
from neural_network_lib.losses import BinaryCrossEntropy, CategoricalCrossentropy


def override_loss_config(loss_type: str, config_path: str = 'neural_network_lib/config/model_config.yaml'):
    """
    Temporarily modifies the loss function configuration.
    Used for CLI override functionality.
    
    Args:
        loss_type (str): 'binary' or 'categorical'
        config_path (str): Path to the configuration file
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        config['loss_function'] = loss_type.lower()
        
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            
        print(f"Temporary configuration: loss_function = '{loss_type}'")
        
    except Exception as e:
        print(f"Error modifying configuration: {e}")


class LossManager:
    """
    Modular loss function manager.
    Automatically handles adaptation between BCE and CCE based on configuration.
    """
    
    def __init__(self, config_path: str = 'neural_network_lib/config/model_config.yaml'):
        """
        Initialize the loss manager according to configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.loss_type = self._load_loss_config()
        self.loss_function = self._create_loss_function()
    
    def _load_loss_config(self) -> str:
        """Load the loss function configuration."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('loss_function', 'binary').lower()
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}, using default 'binary'")
            return 'binary'
        except Exception as e:
            print(f"Error loading config: {e}, using default 'binary'")
            return 'binary'
    
    def _create_loss_function(self):
        """Create the appropriate loss function instance."""
        if self.loss_type == 'binary':
            return BinaryCrossEntropy()
        elif self.loss_type == 'categorical':
            return CategoricalCrossentropy()
        else:
            print(f"Unknown loss type: {self.loss_type}, defaulting to binary")
            return BinaryCrossEntropy()
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray, 
                    last_layer_type: str = None) -> float:
        """
        Compute loss with automatic adaptation to output type.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): True labels
            last_layer_type (str): Type of last layer (for info)
            
        Returns:
            float: Loss value
        """
        if self.loss_type == 'binary':
            return self._compute_binary_loss(predictions, targets, last_layer_type)
        else:
            return self._compute_categorical_loss(predictions, targets)
    
    def _compute_binary_loss(self, predictions: np.ndarray, targets: np.ndarray, 
                           last_layer_type: str = None) -> float:
        """
        Compute Binary Cross Entropy.
        If Softmax + 2 classes → extracts malignant class probability.
        """
        return self.loss_function.forward(predictions, targets)
    
    def _compute_categorical_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Categorical Cross Entropy.
        Automatically converts targets to one-hot if necessary.
        """
        targets = np.array(targets).ravel()
        
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            n_samples = targets.size
            n_classes = predictions.shape[1]
            
            unique_targets = np.unique(targets)
            if not all(0 <= t < n_classes for t in unique_targets):
                print(f"Warning: target values {unique_targets} outside range [0, {n_classes-1}]")
            
            y_one_hot = np.zeros((n_samples, n_classes))
            valid_indices = (targets >= 0) & (targets < n_classes)
            y_one_hot[np.arange(n_samples)[valid_indices], targets[valid_indices].astype(int)] = 1
            
            print(f"Targets shape: {targets.shape}, unique values: {np.unique(targets)}")
            print(f"One-hot shape: {y_one_hot.shape}, sum per sample: {y_one_hot.sum(axis=1)[:5]}")
            
            return self.loss_function.forward(predictions, y_one_hot)
        else:
            return self.loss_function.forward(predictions, targets)
    
    def get_loss_info(self) -> dict:
        """Return information about the loss configuration."""
        return {
            'loss_type': self.loss_type,
            'loss_class': self.loss_function.__class__.__name__,
            'description': self._get_loss_description()
        }
    
    def _get_loss_description(self) -> str:
        """Return a description of the current configuration."""
        if self.loss_type == 'binary':
            return "Binary Cross Entropy - automatically extracts malignant class probability if Softmax"
        else:
            return "Categorical Cross Entropy - uses all class probabilities"
