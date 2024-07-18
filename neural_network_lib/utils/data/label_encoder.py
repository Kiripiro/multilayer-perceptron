import numpy as np
import pandas as pd

def label_encoder(file_path, target_column, positive_class, negative_class):
    """
    This function loads data from a CSV file located at the specified file path, reads the data using pandas, 
    separates the features (X) from the target variable (y), and encodes the target variable values 
    to 0 and 1 based on specified positive and negative classes.

    Args:
        file_path (str): The path to the CSV file containing the data.
        target_column (str): The name of the column containing the target variable.
        positive_class (str): The value in the target column to be encoded as 1.
        negative_class (str): The value in the target column to be encoded as 0.

    Returns:
        X (numpy.ndarray): A 2D numpy array containing the features of the data.
        y (numpy.ndarray): A 1D numpy array containing the encoded target variable values.
    """
    data = pd.read_csv(file_path)
    
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    
    y = np.where(y == positive_class, 1, 0)
    
    return X, y