from neural_network_lib.utils.model.build_model import build_model
from neural_network_lib.utils.model.save_model import save_model
from neural_network_lib.utils.model.loss_manager import LossManager
from neural_network_lib.optimizers.sgd import SGD, SGDMomentum
from neural_network_lib.optimizers.adam import Adam
from neural_network_lib.metrics.metrics import precision_recall_f1_score
from colorama import Fore, Style
import numpy as np
from tqdm import tqdm
from typing import Union
import os

def train(
    X_train, y_train,
    X_val, y_val,
    epochs: int,
    optimizer: str,
    learning_rate: float,
    batch_size: int = 32,
    early_stopping: Union[bool, None] = False,
    patience: int = 3,
    scaler_stats=None,
):
    """
    Train an MLP on (X_train, y_train) and evaluate on (X_val, y_val).

    Parameters
    ----------
    epochs : int
    optimizer : {'sgd', 'sgdMomentum', 'adam'}
    learning_rate : float
    batch_size : int
    early_stopping : bool
    patience : int
    scaler_stats : tuple(mean, std) or None

    Returns
    -------
    model  : trained model (best state if early stopping)
    history: dict containing curves 'train_loss', 'val_loss', etc.
    """

    print(f"{Fore.BLUE}Training parameters:{Style.RESET_ALL}")
    print(f"  - Epochs        : {epochs}")
    print(f"  - Optimizer     : {optimizer}")
    print(f"  - Learning Rate : {learning_rate}")
    print(f"  - Batch Size    : {batch_size}")
    print(f"  - EarlyStopping : {'Enabled' if early_stopping else 'Disabled'}")
    if early_stopping:
        print(f"  - Patience      : {patience}")

    model = build_model()

    if optimizer == "sgd":
        opt = SGD(learning_rate)
    elif optimizer == "sgdMomentum":
        opt = SGDMomentum(learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer}' not recognised.")

    loss_manager = LossManager()
    loss_info = loss_manager.get_loss_info()
    print(f"{Fore.CYAN}Loss function: {loss_info['loss_class']} ({loss_info['loss_type']}){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{loss_info['description']}{Style.RESET_ALL}")
    
    model.compile(optimizer=opt, loss=loss_manager.loss_function)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    best_weights, best_biases = None, None
    patience_counter = 0

    n_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = 0.0, 0.0

        idx = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[idx], y_train[idx]

        with tqdm(total=n_batches, desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for b in range(n_batches):
                start, end = b * batch_size, min((b + 1) * batch_size, X_train.shape[0])
                xb, yb = X_train[start:end], y_train[start:end]

                batch_hist = model.fit(xb, yb, epochs=1)
                batch_loss = batch_hist["loss"][-1]
                train_loss += batch_loss

                y_pred = model.predict(xb)
                y_true = np.array(yb).ravel()
                if y_pred.ndim > 1:
                    y_pred = (y_pred >= 0.5).astype(int).ravel()
                batch_acc = np.mean(y_pred == y_true)
                train_acc += batch_acc

                pbar.update(1)

        train_loss /= n_batches
        train_acc  /= n_batches

        val_loss = model.loss.forward(model.predict(X_val, return_probs=True), y_val)
        precision, recall, f1, _ = precision_recall_f1_score(y_val, model.predict(X_val))
        val_acc = np.mean(model.predict(X_val) == y_val.ravel())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} - "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_weights = [
                    layer.weights.copy() for layer in model.layers if hasattr(layer, "weights")
                ]
                best_biases = [
                    layer.biases.copy() for layer in model.layers if hasattr(layer, "biases")
                ]
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"{Fore.YELLOW}Early stopping at epoch {epoch}{Style.RESET_ALL}")
                    break

    if early_stopping and best_weights is not None:
        w_idx, b_idx = 0, 0
        for layer in model.layers:
            if hasattr(layer, "weights"):
                layer.weights = best_weights[w_idx];  w_idx += 1
            if hasattr(layer, "biases"):
                layer.biases  = best_biases[b_idx];  b_idx += 1
        print(f"{Fore.GREEN}Best model (val_loss={best_val_loss:.4f}) restored.{Style.RESET_ALL}")

    save_model(model)
    if scaler_stats is not None:
        mean, std = scaler_stats
        os.makedirs('data/model', exist_ok=True)
        np.savez('data/model/scaler.npz', mean=mean, std=std)
    print(f"{Fore.CYAN}Model saved to 'data/model'.{Style.RESET_ALL}")

    return model, history
