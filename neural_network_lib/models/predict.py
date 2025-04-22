import numpy as np

def predict(model, X, return_probs=False):
    """
    Predict the class labels or probabilities for input data using a given model.

    Args:
        model (object): The model used for prediction.
        X (array-like): Input data to make predictions on.
        return_probs (bool): If True, return probabilities; if False, return class labels.

    Returns:
        array-like: Predicted class labels or probabilities for the input data.
    """
    probabilities = model.forward(X)
    if return_probs:
        return probabilities
    if probabilities.shape[1] == 2:
        predictions = np.argmax(probabilities, axis=1)
    else:
        predictions = (probabilities > 0.5).astype(int)
    return predictions