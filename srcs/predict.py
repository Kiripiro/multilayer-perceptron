import numpy as np
from neural_network_lib.layers.activations import Softmax
from neural_network_lib.models.predict import predict
from neural_network_lib.utils import load_model, label_encoder, preprocess_data
from neural_network_lib.utils.model.loss_manager import LossManager
from neural_network_lib.metrics import classification_report
from neural_network_lib.visualizer.visualizer import plot_confusion_matrix, plot_activations
from neural_network_lib.losses import CategoricalCrossentropy, BinaryCrossEntropy

def is_binary_case(predictions):
    """
    Determine if the prediction output corresponds to a binary classification scenario.

    This function checks the shape of the provided predictions array to determine if the model 
    returns a single probability per sample (binary case) or multiple scores per sample (multi-class case).
    It accepts both 1-dimensional arrays (n,) as well as 2-dimensional arrays with a single column (n,1) 
    as binary cases, while a 2-dimensional array with more than one column (n,k with k > 1) is considered 
    a multi-class scenario.

    Args:
        predictions (array-like): An array-like object containing model predictions. This object will be 
            converted to a NumPy array and is expected to have one of the following shapes:
              - (n,) for a binary case.
              - (n, 1) for a binary case.
              - (n, k) with k > 1 for a multi-class case.

    Returns:
        bool: Returns True if the predictions array represents a binary classification case 
              (i.e., a single prediction per sample), and False if it represents a multi-class case 
              (i.e., multiple scores per sample).

    Raises:
        ValueError: If the shape of the predictions array does not match any of the expected formats.
    """
    p = np.array(predictions)
    if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
        return True
    if p.ndim == 2 and p.shape[1] > 1:
        return False
    raise ValueError(f"Unexpected prediction shape {p.shape}")

def evaluate_model(test_csv, activations, show_plots=True):
    """
    Evaluate the trained model on the test dataset and print performance metrics.

    Parameters:
    test_csv : str
        The path to the test CSV file containing the test dataset.
    activations : bool
        Whether to plot neuron activations.
    show_plots : bool
        Whether to display evaluation plots. Default is True.

    Returns:
    None
    """
    try:
        X_test, y_test = label_encoder(
            test_csv,
            target_column='diagnosis',
            positive_class='M',
            negative_class='B'
        )
        X_test = preprocess_data(X_test)
        y_test = np.array(y_test).ravel()

        model = load_model()
        last_layer = model.get_last_layer()
        last_layer_name = last_layer.__class__.__name__ if last_layer else "No layers"
        print(f"Last layer type: {last_layer_name}")

        predictions_proba = predict(model, X_test, return_probs=True)
        predictions_labels = np.array(predict(model, X_test, return_probs=False)).ravel()

        loss_manager = LossManager()
        loss_info = loss_manager.get_loss_info()
        
        print(f"\n--- Loss Function Configuration ---")
        print(f"Type: {loss_info['loss_class']} ({loss_info['loss_type']})")
        print(f"Description: {loss_info['description']}")
        
        loss_value = loss_manager.compute_loss(predictions_proba, y_test, last_layer_name)
        print(f"\n{loss_info['loss_class']} Loss: {loss_value:.4f}")
        
        if loss_info['loss_type'] == 'binary' and predictions_proba.ndim > 1 and predictions_proba.shape[1] > 1:
            n_samples = y_test.size
            n_classes = predictions_proba.shape[1] 
            y_test_oh = np.zeros((n_samples, n_classes))
            y_test_oh[np.arange(n_samples), y_test.astype(int)] = 1
            cce_loss = CategoricalCrossentropy().forward(predictions_proba, y_test_oh)
            print(f"Categorical Cross-Entropy Loss (comparison): {cce_loss:.4f}")
        
        elif loss_info['loss_type'] == 'categorical' and predictions_proba.ndim > 1 and predictions_proba.shape[1] > 1:
            bce_loss = BinaryCrossEntropy().forward(predictions_proba, y_test)
            print(f"Binary Cross-Entropy Loss (comparison): {bce_loss:.4f}")

        print(classification_report(y_test, predictions_labels))

        if show_plots:
            plot_confusion_matrix(
                y_test,
                predictions_labels,
                labels=["Benign", "Malignant"],
                labels_values=[0, 1]
            )

        if activations:
            idx = np.random.choice(X_test.shape[0], size=10, replace=False)
            plot_activations(model, X_test[idx])

    except FileNotFoundError:
        print(f"Error: File {test_csv} does not exist.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
