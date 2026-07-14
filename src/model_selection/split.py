"""Train/test split utility."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets.

    Args:
        X: 2D feature matrix (n_samples, n_features).
        y: 1D label array.
        test_size: Proportion of samples to allocate to the test set.
        random_state: Seed for reproducible shuffling.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If test_size is not strictly between 0 and 1.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    n_samples = X.shape[0]
    n_test = int(np.ceil(n_samples * test_size))

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
