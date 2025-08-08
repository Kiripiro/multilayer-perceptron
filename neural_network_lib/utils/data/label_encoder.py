import numpy as np
import pandas as pd

def _load_dataset_with_headers(file_path: str) -> pd.DataFrame:
    """
    Load a dataset and ensure expected headers exist. Supports files without headers
    formatted as: id, diagnosis, then 30 feature columns matching the standard order.
    """
    feature_columns = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
        'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        first_values = first_line.split(',')

        if len(first_values) == 32:
            try:
                int(first_values[0])
                if first_values[1] in ['M', 'B']:
                    df = pd.read_csv(file_path, header=None)
                    df.columns = ['id', 'diagnosis'] + feature_columns
                    return df
            except (ValueError, IndexError):
                pass
    except Exception:
        pass

    return pd.read_csv(file_path)

def label_encoder(file_path, target_column, positive_class, negative_class):
    """
    Load data from CSV, ensure headers exist (auto-infer if missing),
    separate features (X) from target (y), and encode target values to 0/1.

    Args:
        file_path (str): The path to the CSV file containing the data.
        target_column (str): The name of the column containing the target variable.
        positive_class (str): The value in the target column to be encoded as 1.
        negative_class (str): The value in the target column to be encoded as 0.

    Returns:
        X (numpy.ndarray): Features matrix.
        y (numpy.ndarray): Encoded target vector (0/1).
    """
    data = _load_dataset_with_headers(file_path)

    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset columns: {list(data.columns)}")

    columns_to_exclude = [target_column]
    if 'id' in data.columns:
        columns_to_exclude.append('id')

    X = data.drop(columns=columns_to_exclude).values
    y = data[target_column].values

    y = np.where(y == positive_class, 1, 0)

    return X, y