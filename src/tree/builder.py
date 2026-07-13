"""Recursive Decision Tree training algorithm."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.metrics.impurity import compute_impurity
from src.tree.node import Node
from src.tree.splitter import SplitResult, find_best_split
from src.tree.tree import Tree


class TreeBuilder:
    """Builds a Decision Tree recursively from training data.

    Attributes:
        criterion: Impurity criterion, 'gini' or 'entropy'.
        max_depth: Maximum tree depth (None for unlimited).
        min_samples_split: Minimum samples required to consider a split.
        min_samples_leaf: Minimum samples required in each resulting leaf.
        min_impurity_decrease: Minimum information gain required to split.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """Initialize the TreeBuilder with stopping hyperparameters.

        Args:
            criterion: Impurity criterion, 'gini' or 'entropy'.
            max_depth: Maximum tree depth (None for unlimited).
            min_samples_split: Minimum samples required to consider a split.
            min_samples_leaf: Minimum samples required in each leaf.
            min_impurity_decrease: Minimum gain required to accept a split.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def build(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> Tree:
        """Build a full Decision Tree from training data.

        Args:
            X: 2D feature matrix (n_samples, n_features).
            y: 1D integer label array.
            n_classes: Total number of distinct classes.

        Returns:
            Trained Tree instance.
        """
        root = self._build_node(X, y, depth=0, n_classes=n_classes)
        return Tree(root)

    def _compute_leaf_value(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """Compute class probability distribution for a leaf.

        Args:
            y: 1D integer label array at the node.
            n_classes: Total number of distinct classes.

        Returns:
            Array of class probabilities of length n_classes.
        """
        counts = np.bincount(y, minlength=n_classes)
        return counts / counts.sum()

    def _is_stopping_condition_met(
        self, y: np.ndarray, depth: int, impurity: float
    ) -> bool:
        """Check whether growth should stop at the current node.

        Args:
            y: 1D integer label array at the node.
            depth: Current depth in the tree.
            impurity: Impurity value at the node.

        Returns:
            True if any stopping criterion is met.
        """
        if impurity == 0.0:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if y.size < self.min_samples_split:
            return True
        return False

    def _build_node(
        self, X: np.ndarray, y: np.ndarray, depth: int, n_classes: int
    ) -> Node:
        """Recursively build a subtree rooted at the current node.

        Args:
            X: Feature matrix at the current node.
            y: Label array at the current node.
            depth: Current depth in the tree.
            n_classes: Total number of distinct classes.

        Returns:
            A leaf or split Node.
        """
        impurity = compute_impurity(y, self.criterion)

        if self._is_stopping_condition_met(y, depth, impurity):
            return Node.create_leaf(
                value=self._compute_leaf_value(y, n_classes),
                n_samples=y.size,
                impurity=impurity,
            )

        split: Optional[SplitResult] = find_best_split(
            X, y, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf
        )

        if split is None or split.gain < self.min_impurity_decrease:
            return Node.create_leaf(
                value=self._compute_leaf_value(y, n_classes),
                n_samples=y.size,
                impurity=impurity,
            )

        left_child = self._build_node(
            X[split.left_mask], y[split.left_mask], depth + 1, n_classes
        )
        right_child = self._build_node(
            X[split.right_mask], y[split.right_mask], depth + 1, n_classes
        )

        return Node.create_split(
            feature_index=split.feature_index,
            threshold=split.threshold,
            left=left_child,
            right=right_child,
            n_samples=y.size,
            impurity=impurity,
        )