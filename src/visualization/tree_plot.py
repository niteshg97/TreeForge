"""Visualization utilities for TreeForge Decision Trees."""

from __future__ import annotations

from graphviz import Digraph

from src.tree.node import Node
from src.tree.tree import Tree


class TreePlotter:
    """Visualize a TreeForge decision tree using Graphviz."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        """Initialize the Graphviz plotter.

        Args:
            feature_names: Optional list of feature names.
        """
        self.graph = Digraph(
            "TreeForge",
            format="png",
            graph_attr={
                "rankdir": "TB",
                "dpi": "180",
                "nodesep": "0.3",
                "ranksep": "0.4",
            },
        )

        self.feature_names = feature_names

    def build(self, tree: Tree) -> Digraph:
        """Build a Graphviz graph from a Tree."""
        if tree.root is not None:
            self._add_node(tree.root, "0")
        return self.graph

    def _add_node(self, node: Node, node_id: str) -> None:
        """Recursively add nodes and edges."""

        if node.is_leaf():
            probabilities = ", ".join(f"{p:.2f}" for p in node.value)

            predicted_class = int(node.value.argmax())

            label = (
                f"Leaf\n"
                f"Predict = {predicted_class}\n"
                f"Samples = {node.n_samples}\n"
                f"Impurity = {node.impurity:.3f}\n"
                f"Prob = [{probabilities}]"
            )

            self.graph.node(
                node_id,
                label,
                shape="box",
                style="filled,rounded",
                fillcolor="#D5F5E3",
                color="#1E8449",
                fontsize="11",
            )
            return

        if self.feature_names is not None:
            feature = self.feature_names[node.feature_index]
        else:
            feature = f"Feature {node.feature_index}"

        label = (
            f"{feature}\n"
            f"≤ {node.threshold:.3f}\n"
            f"Samples = {node.n_samples}\n"
            f"Impurity = {node.impurity:.3f}"
        )

        self.graph.node(
            node_id,
            label,
            shape="ellipse",
            style="filled",
            fillcolor="#D6EAF8",
            color="#2874A6",
            fontsize="11",
        )

        left_id = node_id + "L"
        right_id = node_id + "R"

        self._add_node(node.left, left_id)
        self._add_node(node.right, right_id)

        self.graph.edge(node_id, left_id, label="True")
        self.graph.edge(node_id, right_id, label="False")


def save_tree_plot(
    tree: Tree,
    filename: str = "assets/images/decision_tree",
    feature_names: list[str] | None = None,
) -> str:
    """Save the decision tree as a PNG image.

    Args:
        tree: Trained Tree object.
        filename: Output filename without extension.
        feature_names: Optional list of feature names.

    Returns:
        Path to generated PNG image.
    """
    plotter = TreePlotter(feature_names)
    graph = plotter.build(tree)
    return graph.render(filename, cleanup=True)