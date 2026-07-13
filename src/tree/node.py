"""Core Node data structure for the Decision Tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Node:
    """Represents a single node in the Decision Tree.

    A node is either a leaf (holds a predicted class distribution) or an
    internal split node (holds a feature/threshold and two children).

    Attributes:
        feature_index: Index of the feature used to split (None for leaves).
        threshold: Threshold value used to split (None for leaves).
        left: Left child node (samples where feature <= threshold).
        right: Right child node (samples where feature > threshold).
        value: Class probability distribution for leaf nodes (None otherwise).
        n_samples: Number of training samples that reached this node.
        impurity: Impurity value (e.g. Gini/Entropy) at this node.
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[np.ndarray] = None
    n_samples: int = 0
    impurity: float = 0.0

    def is_leaf(self) -> bool:
        """Check whether this node is a leaf node.

        Returns:
            True if the node holds a prediction value, False otherwise.
        """
        return self.value is not None

    @classmethod
    def create_leaf(
        cls, value: np.ndarray, n_samples: int, impurity: float = 0.0
    ) -> "Node":
        """Create a leaf node.

        Args:
            value: Class probability distribution.
            n_samples: Number of samples at this node.
            impurity: Impurity value at this node.

        Returns:
            A leaf Node instance.
        """
        return cls(value=value, n_samples=n_samples, impurity=impurity)

    @classmethod
    def create_split(
        cls,
        feature_index: int,
        threshold: float,
        left: "Node",
        right: "Node",
        n_samples: int,
        impurity: float = 0.0,
    ) -> "Node":
        """Create an internal split node.

        Args:
            feature_index: Index of the feature to split on.
            threshold: Threshold value for the split.
            left: Left child node.
            right: Right child node.
            n_samples: Number of samples at this node.
            impurity: Impurity value at this node.

        Returns:
            An internal Node instance.
        """
        return cls(
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
            n_samples=n_samples,
            impurity=impurity,
        )