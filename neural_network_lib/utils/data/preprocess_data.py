import numpy as np
from neural_network_lib.utils.data.one_hot_encoder import one_hot_encoder

def preprocess_data(X, y=None, mean=None, std=None, return_stats=False):
    """
    Standardize the feature matrix X and automatically encode y for multiclass scenarios.

    Args:
        X (array-like, shape (n_samples, n_features)):
            Feature matrix.
        y (array-like, shape (n_samples,), optional):
            Label vector.
        mean (array-like, shape (n_features,), optional):
            Pre-computed mean values for standardization. If None, computed from X.
        std (array-like, shape (n_features,), optional):
            Pre-computed standard deviation values for standardization. If None, computed from X.
        return_stats (bool, optional):
            Whether to return the computed mean and std values. Default is False.

    Returns:
        X_proc (array-like, shape (n_samples, n_features)):
            The standardized feature matrix.
        y_proc (array-like or None):
            - None if y is None,
            - Binary labels (array, shape (n_samples, 1)) for binary classification,
            - One-hot encoded array (shape (n_samples, n_classes)) for multiclass classification.
        mean (array-like, shape (n_features,)):
            The mean values used for standardization (only if return_stats=True).
        std (array-like, shape (n_features,)):
            The standard deviation values used for standardization (only if return_stats=True).
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
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
            raise ValueError("Binary labels expected as 0/1, found other values.")
    if return_stats:
        return X_proc, y_proc, mean, std
    return X_proc, y_proc
