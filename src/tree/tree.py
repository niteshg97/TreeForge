"""Tree container wrapping the root Node with structural utilities."""

from __future__ import annotations

from typing import Optional

from src.tree.node import Node


class Tree:
    """Container for a Decision Tree structure.

    Attributes:
        root: Root node of the tree (None if untrained/empty).
    """

    def __init__(self, root: Optional[Node] = None) -> None:
        """Initialize the Tree.

        Args:
            root: Root node of the tree.
        """
        self.root = root

    def get_depth(self) -> int:
        """Compute the depth of the tree.

        Returns:
            Maximum depth (0 for a single-leaf or empty tree).
        """
        return self._depth(self.root)

    def _depth(self, node: Optional[Node]) -> int:
        """Recursively compute depth from a given node.

        Args:
            node: Current node.

        Returns:
            Depth of the subtree rooted at node.
        """
        if node is None or node.is_leaf():
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def get_n_nodes(self) -> int:
        """Count total number of nodes (leaves + internal).

        Returns:
            Total node count.
        """
        return self._count_nodes(self.root)

    def _count_nodes(self, node: Optional[Node]) -> int:
        """Recursively count nodes from a given node.

        Args:
            node: Current node.

        Returns:
            Node count of the subtree rooted at node.
        """
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def get_n_leaves(self) -> int:
        """Count total number of leaf nodes.

        Returns:
            Leaf node count.
        """
        return self._count_leaves(self.root)

    def _count_leaves(self, node: Optional[Node]) -> int:
        """Recursively count leaf nodes from a given node.

        Args:
            node: Current node.

        Returns:
            Leaf count of the subtree rooted at node.
        """
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
