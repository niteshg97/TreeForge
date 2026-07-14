"""Public sklearn-style Decision Tree Classifier API."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.tree.builder import TreeBuilder
from src.tree.node import Node
from src.tree.tree import Tree


class NotFittedError(Exception):
    """Raised when predict/predict_proba is called before fit."""


class DecisionTreeClassifier:
    """A from-scratch Decision Tree classifier with an sklearn-like API.

    Attributes:
        criterion: Impurity criterion, 'gini' or 'entropy'.
        max_depth: Maximum tree depth (None for unlimited).
        min_samples_split: Minimum samples required to consider a split.
        min_samples_leaf: Minimum samples required in each resulting leaf.
        min_impurity_decrease: Minimum information gain required to split.
        tree_: Fitted Tree instance (set after calling fit).
        classes_: Sorted array of original class labels seen during fit.
        n_classes_: Number of distinct classes seen during fit.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """Initialize the classifier with hyperparameters.

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

        self.tree_: Optional[Tree] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate shapes and types of input arrays.

        Args:
            X: 2D feature matrix.
            y: 1D label array (optional, validated only if provided).

        Raises:
            ValueError: If X is not 2D, or y's length mismatches X's rows.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        if y is not None and y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have matching sample counts: "
                f"{X.shape[0]} != {y.shape[0]}."
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Train the Decision Tree on the given data.

        Args:
            X: 2D feature matrix (n_samples, n_features).
            y: 1D array of class labels (any hashable/orderable type).

        Returns:
            The fitted classifier instance (self).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self._validate_input(X, y)

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        y_encoded = np.searchsorted(self.classes_, y)

        builder = TreeBuilder(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        self.tree_ = builder.build(X, y_encoded, n_classes=self.n_classes_)
        return self

    def _check_is_fitted(self) -> None:
        """Ensure the classifier has been fitted.

        Raises:
            NotFittedError: If fit() has not been called yet.
        """
        if self.tree_ is None:
            raise NotFittedError(
                "This DecisionTreeClassifier instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

    def _traverse(self, x: np.ndarray, node: Node) -> np.ndarray:
        """Route a single sample through the tree to reach a leaf.

        Args:
            x: 1D feature vector for a single sample.
            node: Current node being evaluated.

        Returns:
            Class probability distribution stored at the reached leaf.
        """
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each sample.

        Args:
            X: 2D feature matrix (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.

        Raises:
            NotFittedError: If called before fit().
        """
        self._check_is_fitted()
        X = np.asarray(X)
        self._validate_input(X)
        return np.array([self._traverse(x, self.tree_.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each sample.

        Args:
            X: 2D feature matrix (n_samples, n_features).

        Returns:
            Array of predicted labels in the original label space.

        Raises:
            NotFittedError: If called before fit().
        """
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]
