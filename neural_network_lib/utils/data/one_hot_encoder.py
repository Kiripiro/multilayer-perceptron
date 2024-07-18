import numpy as np
from numpy import ndarray

def one_hot_encoder(labels) -> ndarray:
    """
    One-hot encodes the given label array.

    Args:
    labels : array-like, shape (n_samples,)
        The target labels as integers or strings.

    Returns:
    encoded_labels : array-like, shape (n_samples, n_classes)
        The one-hot encoded label matrix.
    """
    classes = np.unique(labels)
    eye_matrix = np.eye(len(classes))
    class_indices = {value: idx for idx, value in enumerate(classes)}
    encoded_labels = eye_matrix[np.vectorize(class_indices.get)(labels)]
    return encoded_labels
