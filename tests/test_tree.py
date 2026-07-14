"""Unit tests for src.tree.tree module."""

import numpy as np

from src.tree.node import Node
from src.tree.tree import Tree


def test_empty_tree_metrics() -> None:
    tree = Tree()
    assert tree.get_depth() == 0
    assert tree.get_n_nodes() == 0
    assert tree.get_n_leaves() == 0


def test_single_leaf_tree_metrics() -> None:
    root = Node.create_leaf(value=np.array([1.0, 0.0]), n_samples=10)
    tree = Tree(root)
    assert tree.get_depth() == 0
    assert tree.get_n_nodes() == 1
    assert tree.get_n_leaves() == 1


def test_two_level_tree_metrics() -> None:
    left = Node.create_leaf(value=np.array([1.0, 0.0]), n_samples=5)
    right = Node.create_leaf(value=np.array([0.0, 1.0]), n_samples=5)
    root = Node.create_split(
        feature_index=0, threshold=0.5, left=left, right=right, n_samples=10
    )
    tree = Tree(root)
    assert tree.get_depth() == 1
    assert tree.get_n_nodes() == 3
    assert tree.get_n_leaves() == 2


def test_three_level_tree_metrics() -> None:
    ll = Node.create_leaf(value=np.array([1.0, 0.0]), n_samples=2)
    lr = Node.create_leaf(value=np.array([0.0, 1.0]), n_samples=3)
    left = Node.create_split(
        feature_index=1, threshold=0.2, left=ll, right=lr, n_samples=5
    )
    right = Node.create_leaf(value=np.array([0.0, 1.0]), n_samples=5)
    root = Node.create_split(
        feature_index=0, threshold=0.5, left=left, right=right, n_samples=10
    )
    tree = Tree(root)
    assert tree.get_depth() == 2
    assert tree.get_n_nodes() == 5
    assert tree.get_n_leaves() == 3
