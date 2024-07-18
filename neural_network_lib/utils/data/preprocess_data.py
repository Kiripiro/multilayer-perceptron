import numpy as np
from neural_network_lib.utils.data import one_hot_encoder

def preprocess_data(X, y=None, encode_labels=False):
    """
    Standardizes the feature matrix X and optionally one-hot encodes the target vector y.

    Args:
    X : array-like, shape (n_samples, n_features)
        The input feature matrix.
    y : array-like, shape (n_samples,), optional
        The target vector. If encode_labels is True, y will be one-hot encoded.
    encode_labels : bool, default False
        If True, the target vector y will be one-hot encoded.

    Returns:
    X : array-like, shape (n_samples, n_features)
        The standardized feature matrix.
    y : array-like, shape (n_samples, n_classes), optional
        The one-hot encoded target matrix. Returned only if y is provided and encode_labels is True.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
    X = (X - mean) / std

    if y is not None and encode_labels:
        y = one_hot_encoder(y)
        return X, y
    
    return X
