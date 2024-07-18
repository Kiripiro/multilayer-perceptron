import numpy as np
from neural_network_lib.models.predict import predict
from neural_network_lib.utils import load_model
from neural_network_lib.utils import label_encoder
from neural_network_lib.utils import preprocess_data
from neural_network_lib.metrics import classification_report
from neural_network_lib.visualizer.visualizer import plot_confusion_matrix, plot_activations

def evaluate_model(test_csv, activations):
    """
    Evaluate the trained model on the test dataset and print performance metrics.

    Parameters:
    test_csv : str
        The path to the test CSV file containing the test dataset.

    Returns:
    None
    """
    X_test, y_test = label_encoder(test_csv, target_column='diagnosis', positive_class='M', negative_class='B')
    X_test = preprocess_data(X_test)

    model = load_model()

    predictions = predict(model, X_test)

    print(classification_report(y_test, predictions))
    
    plot_confusion_matrix(y_test, predictions, labels=["Benign", "Malignant"], labels_values=[0, 1])

    sample_index = np.random.choice(X_test.shape[0], size=10, replace=False)
    X_sample = X_test[sample_index]
    if (activations):
        plot_activations(model, X_sample)