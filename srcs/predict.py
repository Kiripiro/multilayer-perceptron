import numpy as np
from neural_network_lib.layers.activations import Softmax
from neural_network_lib.models.predict import predict
from neural_network_lib.utils import load_model, label_encoder, preprocess_data
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

def evaluate_model(test_csv, activations):
    """
    Evaluate the trained model on the test dataset and print performance metrics.

    Parameters:
    test_csv : str
        The path to the test CSV file containing the test dataset.

    Returns:
    None
    """
    try:
        # 1) Chargement et pré‑traitement
        X_test, y_test = label_encoder(
            test_csv,
            target_column='diagnosis',
            positive_class='M',
            negative_class='B'
        )
        X_test = preprocess_data(X_test)
        y_test = np.array(y_test).ravel()  # → vecteur 1‑D de 0/1

        # 2) Chargement du modèle
        model = load_model()
        last_layer = model.get_last_layer()
        print(f"Last layer type: {last_layer.__class__.__name__}" if last_layer else "No layers in the model")

        # 3) Prédictions
        predictions_proba = predict(model, X_test, return_probs=True)
        # on aplatit toujours les labels prédits pour classification_report
        predictions_labels = np.array(predict(model, X_test, return_probs=False)).ravel()

        # 4) Détection du cas binaire vs multiclasses
        binary = is_binary_case(predictions_proba)

        # 5) Calcul de la loss (moyenne)
        if binary:
            loss = BinaryCrossEntropy().forward(predictions_proba, y_test).mean()
            print(f"Binary Cross-Entropy Loss (moyenne) : {loss:.4f}")
        else:
            # on reconstruit y_test en one-hot si nécessaire
            n_samples = y_test.size
            n_classes = predictions_proba.shape[1]
            y_test_oh = np.zeros((n_samples, n_classes))
            y_test_oh[np.arange(n_samples), y_test.astype(int)] = 1
            loss = CategoricalCrossentropy().forward(predictions_proba, y_test_oh).mean()
            print(f"Categorical Cross-Entropy Loss (moyenne) : {loss:.4f}")

        # 6) Rapport de classification
        print(classification_report(y_test, predictions_labels))

        # 7) Matrice de confusion
        plot_confusion_matrix(
            y_test,
            predictions_labels,
            labels=["Benign", "Malignant"],
            labels_values=[0, 1]
        )

        # 8) Visualisation des activations (optionnel)
        if activations:
            idx = np.random.choice(X_test.shape[0], size=10, replace=False)
            plot_activations(model, X_test[idx])

    except FileNotFoundError:
        print(f"Erreur : Le fichier {test_csv} n'existe pas.")
    except Exception as e:
        print(f"Erreur lors de l'évaluation : {e}")
