import pandas as pd
import numpy as np
from typing import Tuple, Union

def train_test_split(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], test_size=0.25, random_state=32) -> Tuple:
    """
    Splits the given arrays into random train and test subsets.
    
    Args:
    X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The target labels.
    test_size : float, int, or None (default is 0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to 0.25.
    random_state : int or None (default is 42)
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        
    Returns:
    X_train, X_test, y_train, y_test : Tuple containing train-test split of inputs.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    assert len(set(train_indices) & set(test_indices)) == 0, "Training and testing sets overlap!"

    X_train = X.iloc[train_indices] if isinstance(X, pd.DataFrame) else X[train_indices]
    X_test = X.iloc[test_indices] if isinstance(X, pd.DataFrame) else X[test_indices]
    y_train = y.iloc[train_indices] if isinstance(y, pd.Series) else y[train_indices]
    y_test = y.iloc[test_indices] if isinstance(y, pd.Series) else y[test_indices]
    
    return X_train, X_test, y_train, y_test
