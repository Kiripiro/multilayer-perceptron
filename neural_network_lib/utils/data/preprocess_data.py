import numpy as np
from neural_network_lib.utils.data.one_hot_encoder import one_hot_encoder

def preprocess_data(X, y=None):
    """
    Standardize the feature matrix X and automatically encode y for multiclass scenarios.

    Args:
        X (array-like, shape (n_samples, n_features)):
            Feature matrix.
        y (array-like, shape (n_samples,), optional):
            Label vector.

    Returns:
        X_proc (array-like, shape (n_samples, n_features)):
            The standardized feature matrix.
        y_proc:
            - None if y is None,
            - Binary labels (array, shape (n_samples, 1)) for binary classification,
            - One-hot encoded array (shape (n_samples, n_classes)) for multiclass classification.
    """
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0)
    std  = np.where(std == 0, 1, std)
    X_proc = (X - mean) / std

    if y is None:
        return X_proc

    unique_vals = np.unique(y)
    n_classes  = unique_vals.shape[0]

    if n_classes > 2:
        y_proc = one_hot_encoder(y)
    else:
        y_arr = np.array(y).reshape(-1, 1)
        if set(np.unique(y_arr)) <= {0, 1}:
            y_proc = y_arr
        else:
            raise ValueError("Labels binaires attendus en 0/1, trouvé d’autres valeurs.")
    
    return X_proc, y_proc
