"""Best-split search logic for the Decision Tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.metrics.impurity import information_gain


@dataclass
class SplitResult:
    """Represents the best split found for a node.

    Attributes:
        feature_index: Index of the feature to split on.
        threshold: Threshold value for the split.
        gain: Information gain achieved by this split.
        left_mask: Boolean mask selecting samples for the left child.
        right_mask: Boolean mask selecting samples for the right child.
    """

    feature_index: int
    threshold: float
    gain: float
    left_mask: np.ndarray
    right_mask: np.ndarray


def get_candidate_thresholds(feature_values: np.ndarray) -> np.ndarray:
    """Generate candidate split thresholds as midpoints of sorted unique values.

    Args:
        feature_values: 1D array of values for a single feature.

    Returns:
        Array of candidate thresholds (empty if fewer than 2 unique values).
    """
    unique_values = np.unique(feature_values)
    if unique_values.size < 2:
        return np.array([])
    return (unique_values[:-1] + unique_values[1:]) / 2.0


def split_dataset(
    feature_values: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split a feature column into left/right boolean masks by threshold.

    Args:
        feature_values: 1D array of values for a single feature.
        threshold: Threshold value to split on.

    Returns:
        Tuple of (left_mask, right_mask) where left is <= threshold.
    """
    left_mask = feature_values <= threshold
    right_mask = ~left_mask
    return left_mask, right_mask


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
    min_samples_leaf: int = 1,
) -> Optional[SplitResult]:
    """Search all features and thresholds for the split with highest gain.

    Args:
        X: 2D feature matrix (n_samples, n_features).
        y: 1D label array.
        criterion: Impurity criterion, 'gini' or 'entropy'.
        min_samples_leaf: Minimum samples required in each resulting leaf.

    Returns:
        Best SplitResult found, or None if no valid split exists.
    """
    n_samples, n_features = X.shape
    best_split: Optional[SplitResult] = None
    best_gain = 0.0

    if n_samples < 2 * min_samples_leaf:
        return None

    for feature_index in range(n_features):
        feature_values = X[:, feature_index]
        thresholds = get_candidate_thresholds(feature_values)

        for threshold in thresholds:
            left_mask, right_mask = split_dataset(feature_values, threshold)

            if (
                left_mask.sum() < min_samples_leaf
                or right_mask.sum() < min_samples_leaf
            ):
                continue

            gain = information_gain(y, y[left_mask], y[right_mask], criterion=criterion)

            if gain > best_gain:
                best_gain = gain
                best_split = SplitResult(
                    feature_index=feature_index,
                    threshold=float(threshold),
                    gain=gain,
                    left_mask=left_mask,
                    right_mask=right_mask,
                )

    return best_split
