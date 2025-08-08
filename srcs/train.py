import os
import pandas as pd
from neural_network_lib.models import train
from neural_network_lib.utils import preprocess_data
from neural_network_lib.utils.model.loss_manager import override_loss_config
from neural_network_lib.visualizer.visualizer import plot_learning_curves
from colorama import Fore, Style


def validate_parameters(epochs, optimizer, learning_rate, early_stopping, patience):
    """
    Validates the parameters for training a neural network model.

    Args:
        epochs (int): The number of epochs for training.
        optimizer (str): The optimizer to use during training ('sgd', 'sgdMomentum', or 'adam').
        learning_rate (float): The learning rate for the optimizer.
        early_stopping (bool): Whether to use early stopping during training.
        patience (int): The number of epochs to wait before early stopping if no improvement.

    Raises:
        ValueError: If any of the parameters are invalid.

    Returns:
        None
    """
    if epochs <= 0 or not isinstance(epochs, int):
        raise ValueError("Number of epochs must be a positive integer.")
    if learning_rate <= 0 or not isinstance(learning_rate, (int, float)):
        raise ValueError("Learning rate must be a positive number.")
    if optimizer not in ['sgd', 'sgdMomentum', 'adam']:
        raise ValueError("Optimizer must be one of 'sgd', 'sgdMomentum', or 'adam'.")
    if early_stopping and (patience <= 0 or not isinstance(patience, int)):
        raise ValueError("Patience must be a positive integer when early stopping is enabled.")


def load_dataset_with_headers(csv_path):
    """
    Loads a dataset and ensures expected headers exist. Supports files without headers
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

    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    first_values = first_line.split(',')

    if len(first_values) == 32:
        try:
            int(first_values[0])
            if first_values[1] in ['M', 'B']:
                df = pd.read_csv(csv_path, header=None)
                df.columns = ['id', 'diagnosis'] + feature_columns
                print(f"{Fore.YELLOW}Headers inferred for {csv_path}{Style.RESET_ALL}")
                return df
        except (ValueError, IndexError):
            pass

    return pd.read_csv(csv_path)


def dataframe_to_xy(df):
    feature_cols = [c for c in df.columns if c not in ['diagnosis', 'id']]
    X = df[feature_cols].values
    y = (df['diagnosis'] == 'M').astype(int).values
    return X, y


def run_training(epochs, optimizer, learning_rate, early_stopping, patience, batch_size=32, loss_function=None, show_plots=True, train_dataset=None, test_dataset=None):
    """
    Runs the training process of a neural network model using the specified parameters.

    Args:
        epochs (int): The number of epochs for training.
        optimizer (str): The optimizer to use during training ('sgd', 'sgdMomentum', or 'adam').
        learning_rate (float): The learning rate for the optimizer.
        early_stopping (bool): Whether to use early stopping during training.
        patience (int): The number of epochs to wait before early stopping if no improvement.
        batch_size (int): Size of each mini-batch for training.
        loss_function (str, optional): Loss function to use ('binary' or 'categorical').
        show_plots (bool): Whether to display training plots. Default is True.
        train_dataset (str, optional): Path to the training dataset CSV file. If None, uses default path.
        test_dataset (str, optional): Path to the test dataset CSV file. If None, uses default path.

    Raises:
        FileNotFoundError: If the specified training or test file paths are not found.

    Output:
        Trained model saved to the 'data/model' directory.
        Learning curves of loss and accuracy plotted.

    Returns:
        None
    """
    validate_parameters(epochs, optimizer, learning_rate, early_stopping, patience)

    if loss_function:
        override_loss_config(loss_function)

    train_csv_path = train_dataset if train_dataset else 'data/train_test/train.csv'
    test_csv_path = test_dataset if test_dataset else 'data/train_test/test.csv'

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f'File {train_csv_path} not found')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f'File {test_csv_path} not found')

    train_df = load_dataset_with_headers(train_csv_path)
    test_df = load_dataset_with_headers(test_csv_path)

    if train_df.empty:
        raise ValueError(f'Training file {train_csv_path} is empty')
    if test_df.empty:
        raise ValueError(f'Test file {test_csv_path} is empty')

    if 'diagnosis' not in train_df.columns:
        raise ValueError(f"Training file {train_csv_path} does not contain the required column: ['diagnosis']")
    if 'diagnosis' not in test_df.columns:
        raise ValueError(f"Test file {test_csv_path} does not contain the required column: ['diagnosis']")

    X_train, y_train = dataframe_to_xy(train_df)
    X_val, y_val = dataframe_to_xy(test_df)

    X_train, y_train, mean, std = preprocess_data(X_train, y_train, return_stats=True)
    X_val, y_val = preprocess_data(X_val, y_val, mean=mean, std=std)

    model, history = train(
        X_train, y_train, X_val, y_val, epochs, optimizer, learning_rate,
        early_stopping=early_stopping, patience=patience, batch_size=batch_size,
        scaler_stats=(mean, std)
    )

    if show_plots:
        plot_learning_curves(history)
