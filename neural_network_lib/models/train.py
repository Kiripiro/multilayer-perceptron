from neural_network_lib.utils import build_model, save_model
from neural_network_lib.optimizers import SGD, SGDMomentum, Adam
from neural_network_lib.losses import BinaryCrossEntropy
import numpy as np
from colorama import Fore, Style
from tqdm import tqdm

def train(X_train, y_train, X_val, y_val,
          epochs, optimizer, learning_rate,
          batch_size, early_stopping=None, patience=1):
    """
    Train a neural network model using the specified optimizer and loss function for a given number of epochs.

    Args:
        X_train (numpy.ndarray): The training input data.
        y_train (numpy.ndarray): The training target data.
        X_val (numpy.ndarray): The validation input data.
        y_val (numpy.ndarray): The validation target data.
        epochs (int): The number of epochs to train the model.
        optimizer (str): The optimizer to use ('sgd', 'sgdMomentum', or 'adam').
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The size of each mini-batch for training. Default is 1.
        early_stopping (bool): Whether to use early stopping based on validation loss.
        patience (int): The number of epochs to wait before early stopping.

    Returns:
        tuple: A tuple containing the trained model and a dictionary of training history.
    """
    print(f"{Fore.BLUE}Training parameters:{Style.RESET_ALL}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Optimizer: {optimizer}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Early Stopping: {'Enabled' if early_stopping else 'Disabled'}")
    if early_stopping:
        print(f"  - Patience: {patience}")
    model = build_model()
    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == 'sgdMomentum':
        optimizer = SGDMomentum(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer}' not recognized.")

    model.compile(optimizer=optimizer, loss=BinaryCrossEntropy())

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc  = 0.0

        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
            for batch in range(num_batches):
                start = batch * batch_size
                end   = min(start + batch_size, X_train.shape[0])
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                batch_history = model.fit(X_batch, y_batch, epochs=1)
                batch_loss = batch_history['loss'][-1]
                train_loss += batch_loss

                y_pred_proba = model.predict(X_batch)
                y_pred_label = (y_pred_proba >= 0.5).astype(int).ravel()
                y_true_label = np.array(y_batch).ravel()
                batch_acc = np.mean(y_pred_label == y_true_label)
                train_acc += batch_acc

                pbar.update(1)

        train_loss /= num_batches
        train_acc  /= num_batches

        val_loss, val_acc = model.evaluate(X_val, y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{epochs} - '
              f'loss: {train_loss:.4f} - acc: {train_acc:.4f} - '
              f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}\n')

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(model, filename='model_weights_biases.npz')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'{Fore.YELLOW}Early stopping at epoch {epoch+1}{Style.RESET_ALL}')
                break

    return model, history
