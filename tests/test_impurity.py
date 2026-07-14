"""Unit tests for src.metrics.impurity module."""

import numpy as np
import pytest

from src.metrics.impurity import (class_probabilities, compute_impurity,
                                  entropy, gini_impurity, information_gain,
                                  weighted_impurity)


def test_class_probabilities_empty() -> None:
    assert class_probabilities(np.array([])).size == 0


def test_class_probabilities_balanced() -> None:
    probs = class_probabilities(np.array([0, 0, 1, 1]))
    np.testing.assert_array_almost_equal(sorted(probs), [0.5, 0.5])


def test_gini_impurity_pure_node() -> None:
    assert gini_impurity(np.array([1, 1, 1, 1])) == 0.0


def test_gini_impurity_balanced_binary() -> None:
    assert gini_impurity(np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_gini_impurity_empty() -> None:
    assert gini_impurity(np.array([])) == 0.0


def test_entropy_pure_node() -> None:
    assert entropy(np.array([0, 0, 0])) == 0.0


def test_entropy_balanced_binary() -> None:
    assert entropy(np.array([0, 1, 0, 1])) == pytest.approx(1.0)


def test_entropy_empty() -> None:
    assert entropy(np.array([])) == 0.0


def test_compute_impurity_invalid_criterion_raises() -> None:
    with pytest.raises(ValueError):
        compute_impurity(np.array([0, 1]), criterion="invalid")


def test_weighted_impurity_equal_splits() -> None:
    left = np.array([0, 0])
    right = np.array([1, 1])
    assert weighted_impurity(left, right, criterion="gini") == 0.0


def test_weighted_impurity_empty_total() -> None:
    assert weighted_impurity(np.array([]), np.array([]), criterion="gini") == 0.0


def test_information_gain_perfect_split() -> None:
    parent = np.array([0, 0, 1, 1])
    left = np.array([0, 0])
    right = np.array([1, 1])
    gain = information_gain(parent, left, right, criterion="gini")
    assert gain == pytest.approx(0.5)


def test_information_gain_no_gain_identical_split() -> None:
    parent = np.array([0, 1, 0, 1])
    left = np.array([0, 1])
    right = np.array([0, 1])
    gain = information_gain(parent, left, right, criterion="entropy")
    assert gain == pytest.approx(0.0)


def test_information_gain_entropy_criterion() -> None:
    parent = np.array([0, 0, 1, 1])
    left = np.array([0, 0])
    right = np.array([1, 1])
    gain = information_gain(parent, left, right, criterion="entropy")
    assert gain == pytest.approx(1.0)
