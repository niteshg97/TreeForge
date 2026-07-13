"""Classification evaluation metrics."""

from __future__ import annotations

import numpy as np

VALID_AVERAGES = ("binary", "macro")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix over the union of true/predicted labels.

    Args:
        y_true: 1D array of ground-truth labels.
        y_pred: 1D array of predicted labels.

    Returns:
        2D array of shape (n_classes, n_classes); rows are true classes,
        columns are predicted classes, ordered by sorted unique labels.
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((labels.size, labels.size), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1

    return matrix


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: 1D array of ground-truth labels.
        y_pred: 1D array of predicted labels.

    Returns:
        Fraction of correctly predicted samples.
    """
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _precision_recall_f1_per_class(
    cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class precision, recall, and F1 from a confusion matrix.

    Args:
        cm: Confusion matrix (rows=true, cols=predicted).

    Returns:
        Tuple of (precision, recall, f1) arrays, one value per class.
    """
    true_positives = np.diag(cm)
    predicted_positives = cm.sum(axis=0)
    actual_positives = cm.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(
            predicted_positives > 0, true_positives / predicted_positives, 0.0
        )
        recall = np.where(
            actual_positives > 0, true_positives / actual_positives, 0.0
        )
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    return precision, recall, f1


def _validate_average(average: str) -> None:
    """Validate the averaging strategy name.

    Args:
        average: Averaging strategy, 'binary' or 'macro'.

    Raises:
        ValueError: If average is not supported.
    """
    if average not in VALID_AVERAGES:
        raise ValueError(
            f"Unsupported average '{average}'. Use one of {VALID_AVERAGES}."
        )


def precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary"
) -> float:
    """Compute precision score.

    Args:
        y_true: 1D array of ground-truth labels.
        y_pred: 1D array of predicted labels.
        average: 'binary' (positive class = max label) or 'macro'.

    Returns:
        Precision score.
    """
    _validate_average(average)
    cm = confusion_matrix(y_true, y_pred)
    precision, _, _ = _precision_recall_f1_per_class(cm)
    if average == "macro":
        return float(np.mean(precision))
    return float(precision[-1])


def recall_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary"
) -> float:
    """Compute recall score.

    Args:
        y_true: 1D array of ground-truth labels.
        y_pred: 1D array of predicted labels.
        average: 'binary' (positive class = max label) or 'macro'.

    Returns:
        Recall score.
    """
    _validate_average(average)
    cm = confusion_matrix(y_true, y_pred)
    _, recall, _ = _precision_recall_f1_per_class(cm)
    if average == "macro":
        return float(np.mean(recall))
    return float(recall[-1])


def f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary"
) -> float:
    """Compute F1 score.

    Args:
        y_true: 1D array of ground-truth labels.
        y_pred: 1D array of predicted labels.
        average: 'binary' (positive class = max label) or 'macro'.

    Returns:
        F1 score.
    """
    _validate_average(average)
    cm = confusion_matrix(y_true, y_pred)
    _, _, f1 = _precision_recall_f1_per_class(cm)
    if average == "macro":
        return float(np.mean(f1))
    return float(f1[-1])