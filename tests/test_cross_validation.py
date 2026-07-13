"""Unit tests for src.model_selection.cross_validation module."""

import numpy as np
import pytest

from src.model_selection.cross_validation import cross_val_score, k_fold_split


def test_k_fold_split_no_overlap_and_full_coverage() -> None:
    folds = list(k_fold_split(n_samples=10, n_splits=5, random_state=1))
    all_val_indices = np.concatenate([val for _, val in folds])
    assert sorted(all_val_indices) == list(range(10))

    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))


def test_k_fold_split_fold_sizes_balanced() -> None:
    folds = list(k_fold_split(n_samples=10, n_splits=3, random_state=1))
    val_sizes = sorted(len(val) for _, val in folds)
    assert val_sizes == [3, 3, 4]


def test_k_fold_split_invalid_n_splits_raises() -> None:
    with pytest.raises(ValueError):
        list(k_fold_split(n_samples=10, n_splits=1))
    with pytest.raises(ValueError):
        list(k_fold_split(n_samples=5, n_splits=10))


def test_cross_val_score_returns_correct_number_of_folds() -> None:
    X = np.tile(np.arange(20).reshape(20, 1).astype(float), (1, 2))
    y = np.array([0, 1] * 10)
    scores = cross_val_score(X, y, n_splits=4, max_depth=3)
    assert scores.shape == (4,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_cross_val_score_separable_data_high_accuracy() -> None:
    X = np.concatenate([np.zeros((10, 1)), np.ones((10, 1))]).astype(float) * 10
    y = np.array([0] * 10 + [1] * 10)
    scores = cross_val_score(X, y, n_splits=5, max_depth=3, random_state=0)
    assert np.mean(scores) > 0.8