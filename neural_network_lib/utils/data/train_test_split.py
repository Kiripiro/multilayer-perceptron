import pandas as pd
import numpy as np
from typing import Union, Tuple

def train_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: Union[float, int] = 0.25,
    random_state: Union[int, None] = 32,
    stratify: Union[pd.Series, np.ndarray, None] = None,
) -> Tuple:
    """
    Splits X and y into training and test sets, with stratification option.

    Args
    ----
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,)
        Target labels.
    test_size : float or int (default 0.25)
        • float : proportion of dataset to include in test split.  
        • int : absolute number of test samples.
    random_state : int or None
        Random seed for reproducibility.
    stratify : array-like or None
        If provided, preserves the proportion of classes from `stratify`
        (often `y`) in train / test sets.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else X
    y = y.reset_index(drop=True) if isinstance(y, pd.Series) else y

    n_samples = len(X)
    if isinstance(test_size, float):
        test_count = int(round(n_samples * test_size))
    else:
        test_count = int(test_size)
    test_count = max(1, min(test_count, n_samples - 1))

    indices_test, indices_train = [], []

    if stratify is not None:
        stratify = stratify.reset_index(drop=True) if isinstance(stratify, pd.Series) else np.asarray(stratify)
        unique_classes, class_counts = np.unique(stratify, return_counts=True)

        for cls, count in zip(unique_classes, class_counts):
            cls_idx = np.where(stratify == cls)[0]
            np.random.shuffle(cls_idx)

            n_test_cls = int(round(test_count * (count / n_samples)))
            n_test_cls = min(n_test_cls, len(cls_idx) - 1)

            indices_test.extend(cls_idx[:n_test_cls])
            indices_train.extend(cls_idx[n_test_cls:])

    else:
        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)
        indices_test = all_indices[:test_count]
        indices_train = all_indices[test_count:]

    assert len(set(indices_train) & set(indices_test)) == 0, "Train and test set overlap detected."

    def _subset(arr, idx):
        if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
            return arr.iloc[idx].reset_index(drop=True)
        return arr[idx]

    X_train = _subset(X, indices_train)
    X_test  = _subset(X, indices_test)
    y_train = _subset(y, indices_train)
    y_test  = _subset(y, indices_test)

    return X_train, X_test, y_train, y_test
