"""K-fold cross-validation utilities."""

from __future__ import annotations

from typing import Callable, Iterator, Tuple

import numpy as np

from src.evaluation.metrics import accuracy_score
from src.tree.classifier import DecisionTreeClassifier

ScoringFunc = Callable[[np.ndarray, np.ndarray], float]


def k_fold_split(
    n_samples: int, n_splits: int = 5, random_state: int = 42
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate train/validation index folds for k-fold cross-validation.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of folds.
        random_state: Seed for reproducible shuffling.

    Yields:
        Tuples of (train_indices, val_indices) for each fold.

    Raises:
        ValueError: If n_splits < 2 or n_splits > n_samples.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed n_samples.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        yield train_indices, val_indices
        current = stop


def cross_val_score(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: ScoringFunc = accuracy_score,
    random_state: int = 42,
    **classifier_kwargs: object,
) -> np.ndarray:
    """Evaluate a DecisionTreeClassifier using k-fold cross-validation.

    Args:
        X: 2D feature matrix (n_samples, n_features).
        y: 1D label array.
        n_splits: Number of folds.
        scoring: Scoring function taking (y_true, y_pred) -> float.
        random_state: Seed for reproducible fold generation.
        **classifier_kwargs: Hyperparameters passed to DecisionTreeClassifier.

    Returns:
        Array of scores, one per fold.
    """
    scores = []
    for train_idx, val_idx in k_fold_split(X.shape[0], n_splits, random_state):
        clf = DecisionTreeClassifier(**classifier_kwargs)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[val_idx])
        scores.append(scoring(y[val_idx], y_pred))
    return np.array(scores)