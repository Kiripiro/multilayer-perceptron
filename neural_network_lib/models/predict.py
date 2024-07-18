import numpy as np

def predict(model, X):
    """
    Predict the class labels for input data using a given model.

    Args:
        model (object): The model used for prediction.
        X (array-like): Input data to make predictions on.

    Returns:
        array-like: Predicted class labels for the input data.
    """
    probabilities = model.forward(X)
    predictions = np.argmax(probabilities, axis=1)
    return predictions