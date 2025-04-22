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
    Computes precision, recall, F1 for binaire (seuillage) ou multiclasses (argmax).

    Args:
        y_true: array-like, shape (n,) or (n,1) or (n,k>1)
        y_pred: array-like, same shape as y_true raw preds:
                - (n,1) probas sigmoïde
                - (n,k) probas softmax
                - (n,) labels déjà encodés
        labels: optional list of label values

    Returns:
        precision, recall, f1, labels_array
    """
    yt = np.array(y_true)
    yp = np.array(y_pred)

    if yt.ndim > 1 and yt.shape[1] > 1:
        y_true_flat = np.argmax(yt, axis=1)
    else:
        y_true_flat = yt.ravel()

    if yp.ndim > 1 and yp.shape[1] > 1:
        y_pred_flat = np.argmax(yp, axis=1)
    elif yp.ndim > 1 and yp.shape[1] == 1:
        y_pred_flat = (yp >= 0.5).astype(int).ravel()
    else:
        y_pred_flat = yp.ravel()

    if labels is None:
        labels = np.unique(np.concatenate((y_true_flat, y_pred_flat)))
    labels = np.array(labels)
    n = len(labels)
    idx = {lab:i for i,lab in enumerate(labels)}

    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n)
    for t, p in zip(y_true_flat, y_pred_flat):
        if t == p:
            tp[idx[t]] += 1
        else:
            fp[idx[p]] += 1
            fn[idx[t]] += 1

    precision = np.nan_to_num(tp / (tp + fp + 1e-10))
    recall    = np.nan_to_num(tp / (tp + fn + 1e-10))
    f1        = np.nan_to_num(2 * (precision * recall) / (precision + recall + 1e-10))

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
