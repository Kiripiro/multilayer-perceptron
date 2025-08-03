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
        clipped_predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        correct_confidences = np.sum(targets * clipped_predictions, axis=1)
        correct_confidences = np.clip(correct_confidences, 1e-15, 1.0)
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
    Automatically extracts malignant class probability from Softmax output.

    Methods:
        forward(predictions, targets): Computes the binary crossentropy loss.
        backward(predictions, targets): Computes the gradient of the binary crossentropy loss.
    """
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes the binary crossentropy loss.
        If predictions has 2 columns (Softmax output), extracts malignant class probability (index 1).

        Args:
            predictions (numpy.ndarray): The predicted probabilities, shape (N,) or (N,2) for Softmax.
            targets (numpy.ndarray): The true binary labels (0 or 1), shape (N,).
        Returns:
            float: The computed binary crossentropy loss (mean).
        """
        if predictions.ndim > 1 and predictions.shape[1] == 2:
            preds = predictions[:, 1]
        else:
            preds = predictions.squeeze()
        
        targs = targets.squeeze()
        
        preds = np.clip(preds, 1e-7, 1 - 1e-7)

        loss = -(targs * np.log(preds) + (1 - targs) * np.log(1 - preds))
        return np.mean(loss)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the binary crossentropy loss.
        If predictions has 2 columns (Softmax output), computes gradient for both classes.

        Args:
            predictions (numpy.ndarray): The predicted probabilities, shape (N,) or (N,2) for Softmax.
            targets (numpy.ndarray): The true binary labels (0 or 1), shape (N,).
        Returns:
            numpy.ndarray: The gradient of the loss w.r.t. predictions, same shape as input.
        """
        targs = targets.squeeze()
        
        if predictions.ndim > 1 and predictions.shape[1] == 2:
            grad = np.zeros_like(predictions)
            
            grad[:, 0] = predictions[:, 0] - (1 - targs)
            grad[:, 1] = predictions[:, 1] - targs
            
            return grad / targs.size
        else:
            preds = np.clip(predictions.squeeze(), 1e-7, 1 - 1e-7)
            grad = -(targs / preds - (1 - targs) / (1 - preds)) / targs.size
            return grad.reshape(predictions.shape)
