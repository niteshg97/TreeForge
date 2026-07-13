"""Unit tests for src.tree.node module."""

import numpy as np

from src.tree.node import Node


def test_create_leaf_is_leaf() -> None:
    leaf = Node.create_leaf(value=np.array([0.3, 0.7]), n_samples=10)
    assert leaf.is_leaf() is True
    assert leaf.n_samples == 10
    np.testing.assert_array_equal(leaf.value, np.array([0.3, 0.7]))


def test_create_split_is_not_leaf() -> None:
    left = Node.create_leaf(value=np.array([1.0, 0.0]), n_samples=5)
    right = Node.create_leaf(value=np.array([0.0, 1.0]), n_samples=5)
    split = Node.create_split(
        feature_index=2, threshold=1.5, left=left, right=right, n_samples=10
    )
    assert split.is_leaf() is False
    assert split.feature_index == 2
    assert split.threshold == 1.5
    assert split.left is left
    assert split.right is right


def test_default_node_is_leaf_by_default() -> None:
    node = Node()
    assert node.is_leaf() is False
    assert node.value is None