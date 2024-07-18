import os
import pandas as pd
from neural_network_lib.models import train
from neural_network_lib.utils import label_encoder
from neural_network_lib.utils import preprocess_data
from neural_network_lib.utils import save_model
from neural_network_lib.visualizer.visualizer import plot_learning_curves
from colorama import Fore, Style

def run_training(epochs, optimizer, learning_rate, early_stopping, patience, batch_size=32):
    """
    Runs the training process of a neural network model using the specified parameters.

    Args:
        epochs (int): The number of epochs for training.
        optimizer (str): The optimizer to use during training ('sgd', 'sgdMomentum', or 'adam').
        learning_rate (float): The learning rate for the optimizer.
        early_stopping (bool): Whether to use early stopping during training.
        patience (int): The number of epochs to wait before early stopping if no improvement.

    Raises:
        FileNotFoundError: If the specified training or test file paths are not found.

    Output:
        Trained model saved to the 'data/model' directory.
        Learning curves of loss and accuracy plotted.

    Returns:
        None
    """
    train_file_path = 'data/train_test/train.csv'
    test_file_path = 'data/train_test/test.csv'

    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f'File {train_file_path} not found')
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f'File {test_file_path} not found')

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    if train_data.empty:
        raise ValueError(f'Training file {train_file_path} is empty')
    if test_data.empty:
        raise ValueError(f'Test file {test_file_path} is empty')
    
    required_columns = ['diagnosis']
    if not all(col in train_data.columns for col in required_columns):
        raise ValueError(f'Training file {train_file_path} does not contain the required columns: {required_columns}')
    if not all(col in test_data.columns for col in required_columns):
        raise ValueError(f'Test file {test_file_path} does not contain the required columns: {required_columns}')


    X_train, y_train = label_encoder(train_file_path, target_column='diagnosis', positive_class='M', negative_class='B')
    X_val, y_val = label_encoder(test_file_path, target_column='diagnosis', positive_class='M', negative_class='B')

    X_train, y_train = preprocess_data(X_train, y_train, encode_labels=True)
    X_val, y_val = preprocess_data(X_val, y_val, encode_labels=True)

    model, history = train(
        X_train, y_train, X_val, y_val, epochs, optimizer, learning_rate,
        early_stopping=early_stopping, patience=patience, batch_size=batch_size
    )

    save_model(model)
    print(f"{Fore.CYAN}Model has been saved to data/model directory.{Style.RESET_ALL}\n")
    plot_learning_curves(history)
