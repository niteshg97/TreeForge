"""End-to-end training pipeline: load data, train, evaluate, report."""

from __future__ import annotations

import numpy as np

from src.data.loader import load_and_process
from src.evaluation.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.model_selection.split import train_test_split
from src.tree.classifier import DecisionTreeClassifier

TARGET_COLUMN = "class"


def run_training_pipeline(
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Run the full data-to-evaluation pipeline for TreeForge.

    Args:
        max_depth: Maximum depth of the Decision Tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required in each leaf.
        criterion: Impurity criterion, 'gini' or 'entropy'.
        test_size: Proportion of data reserved for testing.
        random_state: Seed for reproducible splitting.

    Returns:
        Dictionary of evaluation metrics.
    """
    df = load_and_process(persist=False)
    X = df.drop(columns=[TARGET_COLUMN]).to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1": f1_score(y_test, y_pred, average="binary"),
        "tree_depth": clf.tree_.get_depth(),
        "n_nodes": clf.tree_.get_n_nodes(),
        "n_leaves": clf.tree_.get_n_leaves(),
    }

    print("=== TreeForge Training Report ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


if __name__ == "__main__":
    run_training_pipeline()