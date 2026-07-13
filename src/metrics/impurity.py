"""Impurity measures and information gain for Decision Tree splitting."""

from __future__ import annotations

from typing import Callable

import numpy as np

VALID_CRITERIA = ("gini", "entropy")


def class_probabilities(y: np.ndarray) -> np.ndarray:
    """Compute class proportions from a label array.

    Args:
        y: 1D array of class labels.

    Returns:
        Array of class proportions (empty array if y is empty).
    """
    if y.size == 0:
        return np.array([])
    _, counts = np.unique(y, return_counts=True)
    return counts / y.size


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity of a label array.

    Args:
        y: 1D array of class labels.

    Returns:
        Gini impurity value (0.0 for empty or pure arrays).
    """
    probs = class_probabilities(y)
    if probs.size == 0:
        return 0.0
    return float(1.0 - np.sum(probs**2))


def entropy(y: np.ndarray) -> float:
    """Compute Shannon entropy of a label array.

    Args:
        y: 1D array of class labels.

    Returns:
        Entropy value in bits (0.0 for empty or pure arrays).
    """
    probs = class_probabilities(y)
    if probs.size == 0:
        return 0.0
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


_CRITERIA_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "gini": gini_impurity,
    "entropy": entropy,
}


def compute_impurity(y: np.ndarray, criterion: str = "gini") -> float:
    """Dispatch impurity computation based on criterion name.

    Args:
        y: 1D array of class labels.
        criterion: Either 'gini' or 'entropy'.

    Returns:
        Computed impurity value.

    Raises:
        ValueError: If criterion is not supported.
    """
    if criterion not in VALID_CRITERIA:
        raise ValueError(
            f"Unsupported criterion '{criterion}'. Use one of {VALID_CRITERIA}."
        )
    return _CRITERIA_FUNCS[criterion](y)


def weighted_impurity(
    left_y: np.ndarray, right_y: np.ndarray, criterion: str = "gini"
) -> float:
    """Compute sample-weighted impurity of two child splits.

    Args:
        left_y: Labels in the left split.
        right_y: Labels in the right split.
        criterion: Either 'gini' or 'entropy'.

    Returns:
        Weighted impurity across both splits.
    """
    n_total = left_y.size + right_y.size
    if n_total == 0:
        return 0.0
    left_weight = left_y.size / n_total
    right_weight = right_y.size / n_total
    return (
        left_weight * compute_impurity(left_y, criterion)
        + right_weight * compute_impurity(right_y, criterion)
    )


def information_gain(
    parent_y: np.ndarray,
    left_y: np.ndarray,
    right_y: np.ndarray,
    criterion: str = "gini",
) -> float:
    """Compute information gain from splitting parent into left/right.

    Args:
        parent_y: Labels at the parent node before splitting.
        left_y: Labels in the left split.
        right_y: Labels in the right split.
        criterion: Either 'gini' or 'entropy'.

    Returns:
        Information gain (parent impurity minus weighted child impurity).
    """
    parent_impurity = compute_impurity(parent_y, criterion)
    child_impurity = weighted_impurity(left_y, right_y, criterion)
    return parent_impurity - child_impurity 