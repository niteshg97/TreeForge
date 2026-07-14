"""Unit tests for src.tree.builder module."""

import numpy as np
import pytest

from src.tree.builder import TreeBuilder


@pytest.fixture
def separable_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_build_perfectly_separable_data(
    separable_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = separable_data
    builder = TreeBuilder(criterion="gini")
    tree = builder.build(X, y, n_classes=2)

    assert tree.get_depth() == 1
    assert tree.get_n_leaves() == 2
    assert tree.root.feature_index == 0
    assert tree.root.threshold == pytest.approx(2.5)


def test_build_pure_node_creates_single_leaf() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    builder = TreeBuilder()
    tree = builder.build(X, y, n_classes=2)

    assert tree.get_depth() == 0
    assert tree.get_n_nodes() == 1
    np.testing.assert_array_almost_equal(tree.root.value, [0.0, 1.0])


def test_build_respects_max_depth(
    separable_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = separable_data
    builder = TreeBuilder(max_depth=0)
    tree = builder.build(X, y, n_classes=2)

    assert tree.get_depth() == 0
    assert tree.root.is_leaf() is True


def test_build_respects_min_samples_split(
    separable_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = separable_data
    builder = TreeBuilder(min_samples_split=5)
    tree = builder.build(X, y, n_classes=2)

    assert tree.get_depth() == 0
    assert tree.root.is_leaf() is True


def test_build_respects_min_impurity_decrease(
    separable_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = separable_data
    builder = TreeBuilder(min_impurity_decrease=1.0)
    tree = builder.build(X, y, n_classes=2)

    assert tree.root.is_leaf() is True


def test_build_leaf_value_sums_to_one() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1])
    builder = TreeBuilder(min_samples_split=10)
    tree = builder.build(X, y, n_classes=2)

    assert tree.root.value.sum() == pytest.approx(1.0)
    np.testing.assert_array_almost_equal(tree.root.value, [1 / 3, 2 / 3])


def test_build_multi_class_leaf_value() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2])
    builder = TreeBuilder(min_samples_split=10)
    tree = builder.build(X, y, n_classes=3)

    assert tree.root.value.size == 3
    np.testing.assert_array_almost_equal(tree.root.value, [1 / 3, 1 / 3, 1 / 3])
