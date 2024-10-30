import numpy as np

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Computes the confusion matrix for a classification task where the true and predicted labels are provided as strings.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels of the data as strings.
    - y_pred: array-like of shape (n_samples,) - Predicted labels of the data as strings.
    - labels: array-like of shape (n_classes,), optional - List of labels as strings. If not provided,
              labels used are the sorted list of unique labels in both y_true and y_pred.

    Returns:
    - cm: ndarray of shape (n_classes, n_classes) - Confusion matrix.
    - labels: ndarray of shape (n_classes,) - Labels used to index the matrix.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.array(labels)

    num_classes = len(labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    label_to_index = {label: index for index, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            true_index = label_to_index[true]
            pred_index = label_to_index[pred]
            cm[true_index][pred_index] += 1
        else:
            raise ValueError("Encountered labels not in the labels list.")

    return cm, labels

def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy score.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels of the data.
    - y_pred: array-like of shape (n_samples,) - Predicted labels of the data.

    Returns:
    - accuracy: float - Accuracy score as the proportion of correct predictions.
    """
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)

def precision_recall_f1_score(y_true, y_pred, labels=None):
    """
    Computes the precision, recall, and F1 score for a classification task.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels of the data.
    - y_pred: array-like of shape (n_samples,) - Predicted labels of the data.
    - labels: array-like of shape (n_classes,), optional - List of labels as strings. If not provided,
              labels used are the sorted list of unique labels in both y_true and y_pred.

    Returns:
    - precision: ndarray of shape (n_classes,) - Precision scores for each class.
    - recall: ndarray of shape (n_classes,) - Recall scores for each class.
    - f1: ndarray of shape (n_classes,) - F1 scores for each class.
    - labels: ndarray of shape (n_classes,) - Labels used to index the scores.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(labels)
    
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            if true == pred:
                true_positives[label_to_index[true]] += 1
            else:
                false_positives[label_to_index[pred]] += 1
                false_negatives[label_to_index[true]] += 1
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    
    return precision, recall, f1, labels

def classification_report(y_true, y_pred):
    """
    Generates a classification report including precision, recall, and F1 score for each class.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels of the data.
    - y_pred: array-like of shape (n_samples,) - Predicted labels of the data.

    Returns:
    - report: str - Formatted classification report.
    """
    precision, recall, f1, labels = precision_recall_f1_score(y_true, y_pred)
    
    header = f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
    report = header + "="*len(header) + "\n"
    
    for label, p, r, f in zip(labels, precision, recall, f1):
        report += f"{str(label):<10} {p:<10.2f} {r:<10.2f} {f:<10.2f}\n"
    
    accuracy = accuracy_score(y_true, y_pred)
    report += f"\nAccuracy: {accuracy * 100:.2f}%\n"
    
    return report
