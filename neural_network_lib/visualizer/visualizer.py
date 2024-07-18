import matplotlib.pyplot as plt
import seaborn as sns
from neural_network_lib.metrics import confusion_matrix
from neural_network_lib.utils import one_hot_encoder

def plot_learning_curves(history):
    """
    Plots the learning curves of loss and accuracy over epochs.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', and 'val_acc'.
    
    Returns:
        None
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_loss(history):
    """
    Plots the loss over time.

    Args:
        history (list): List of loss values over epochs.
    
    Returns:
        None
    """
    plt.plot(history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over time')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, labels_values=None):
    """
    Plots the confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of label names.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 7))
    if (labels_values):
        cm, labels_cm = confusion_matrix(y_true, y_pred, labels_values)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        cm, labels = confusion_matrix(y_true, y_pred, labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_cm, yticklabels=labels_cm)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_activations(model, X_sample):
    """
    Plots the activations of each layer in the model for a given sample.

    Args:
        model (Sequential): The neural network model.
        X_sample (numpy.ndarray): Input sample data.
            
    Returns:
        None
    """
    activations = []
    input = X_sample
    for i, layer in enumerate(model.layers):
        if i == 0:
            continue
        if hasattr(layer, 'forward'):
            output = layer.forward(input)
            activations.append(output)
            input = output

    num_layers = len(activations)
    fig, axes = plt.subplots(nrows=1, ncols=num_layers, figsize=(num_layers * 3, 5))
    if num_layers == 1:
        axes = [axes]

    for i, activation in enumerate(activations):
        ax = axes[i]
        sns.heatmap(activation, annot=False, cmap='viridis', ax=ax, cbar=True, linewidths=0.5, linecolor='black')
        ax.set_title(f'Layer {i+1} Activations', fontsize=14, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
