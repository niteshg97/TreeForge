"""Unit tests for src.tree.splitter module."""

import numpy as np
import pytest

from src.tree.splitter import (
    find_best_split,
    get_candidate_thresholds,
    split_dataset,
)


def test_get_candidate_thresholds_basic() -> None:
    thresholds = get_candidate_thresholds(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_almost_equal(thresholds, [1.5, 2.5])


def test_get_candidate_thresholds_single_unique_value() -> None:
    thresholds = get_candidate_thresholds(np.array([5.0, 5.0, 5.0]))
    assert thresholds.size == 0


def test_split_dataset_masks() -> None:
    feature_values = np.array([1.0, 2.0, 3.0, 4.0])
    left_mask, right_mask = split_dataset(feature_values, 2.5)
    np.testing.assert_array_equal(left_mask, [True, True, False, False])
    np.testing.assert_array_equal(right_mask, [False, False, True, True])


def test_find_best_split_single_feature_perfect_split() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    result = find_best_split(X, y, criterion="gini")
    assert result is not None
    assert result.feature_index == 0
    assert result.threshold == pytest.approx(2.5)
    assert result.gain == pytest.approx(0.5)


def test_find_best_split_multi_feature_picks_highest_gain() -> None:
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 20.0],
            [4.0, 20.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    result = find_best_split(X, y, criterion="gini")
    assert result is not None
    # Both features perfectly separate classes; either could be chosen
    # depending on scan order, but gain must be maximal.
    assert result.gain == pytest.approx(0.5)


def test_find_best_split_no_valid_split_returns_none() -> None:
    X = np.array([[1.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    result = find_best_split(X, y, criterion="gini")
    assert result is None


def test_find_best_split_respects_min_samples_leaf() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    result = find_best_split(X, y, criterion="gini", min_samples_leaf=3)
    assert result is None


def test_find_best_split_pure_labels_returns_none() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, 1, 1])
    result = find_best_split(X, y, criterion="gini")
    assert result is None