import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Probability of the positive class.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1, ROC-AUC and confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).tolist(),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)

        # ROC-AUC requires both classes to be present.
        if len(np.unique(y_true)) == 2:
            results["roc_auc"] = float(
                roc_auc_score(
                y_true,
                np.asarray(y_prob)[:, 1]
                if np.asarray(y_prob).ndim == 2
                else y_prob
            )
            )

    return results
