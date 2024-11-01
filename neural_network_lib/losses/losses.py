import numpy as np

class MeanSquaredError:
    """
    Mean Squared Error (MSE) loss function.

    Methods:
        forward(predictions, targets): Computes the MSE loss.
        backward(predictions, targets): Computes the gradient of the MSE loss.
    """
    
    def forward(self, predictions, targets):
        """
        Computes the Mean Squared Error loss.
        
        Args:
            predictions (numpy.ndarray): The predicted values.
            targets (numpy.ndarray): The true values.
        
        Returns:
            float: The computed MSE loss.
        """
        return np.mean((predictions - targets) ** 2)
    
    def backward(self, predictions, targets):
        """
        Computes the gradient of the Mean Squared Error loss.
        
        Args:
            predictions (numpy.ndarray): The predicted values.
            targets (numpy.ndarray): The true values.
        
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the predictions.
        """
        return 2 * (predictions - targets) / targets.size

class CategoricalCrossentropy:
    """
    Categorical Crossentropy loss function for multi-class classification.

    Methods:
        forward(predictions, targets): Computes the categorical crossentropy loss.
        backward(predictions, targets): Computes the gradient of the categorical crossentropy loss.
    """
    
    def forward(self, predictions, targets):
        """
        Computes the categorical crossentropy loss.
        
        Args:
            predictions (numpy.ndarray): The predicted probabilities for each class.
            targets (numpy.ndarray): The true class labels in one-hot encoded form.
        
        Returns:
            float: The computed categorical crossentropy loss.
        """
        samples = len(predictions)
        clipped_predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(targets * clipped_predictions, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
    
    def backward(self, predictions, targets):
        """
        Computes the gradient of the categorical crossentropy loss.
        
        Args:
            predictions (numpy.ndarray): The predicted probabilities for each class.
            targets (numpy.ndarray): The true class labels in one-hot encoded form.
        
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the predictions.
        """
        samples = len(predictions)
        clipped_predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        gradient = -targets / clipped_predictions
        return gradient / samples

class BinaryCrossEntropy:
    """
    Binary Crossentropy loss function for binary classification.

    Methods:
        forward(predictions, targets): Computes the binary crossentropy loss.
        backward(predictions, targets): Computes the gradient of the binary crossentropy loss.
    """
    
    def forward(self, predictions, targets):
        """
        Computes the binary crossentropy loss.
        
        Args:
            predictions (numpy.ndarray): The predicted probabilities.
            targets (numpy.ndarray): The true binary labels (0 or 1).
        
        Returns:
            float: The computed binary crossentropy loss.
        """
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        loss = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return np.mean(loss)
    
    def backward(self, predictions, targets):
        """
        Computes the gradient of the binary crossentropy loss.
        
        Args:
            predictions (numpy.ndarray): The predicted probabilities.
            targets (numpy.ndarray): The true binary labels (0 or 1).
        
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the predictions.
        """
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        gradient = -(targets / predictions - (1 - targets) / (1 - predictions))
        return gradient / targets.size
