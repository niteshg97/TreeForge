"""Benchmark comparison: TreeForge DecisionTreeClassifier vs sklearn."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from src.data.loader import load_and_process
from src.evaluation.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from src.model_selection.split import train_test_split
from src.tree.classifier import DecisionTreeClassifier as TreeForgeClassifier

TARGET_COLUMN = "class"


@dataclass
class BenchmarkResult:
    """Holds benchmark metrics for a single model.

    Attributes:
        name: Model name/label.
        accuracy: Accuracy on the test set.
        precision: Precision on the test set (binary average).
        recall: Recall on the test set (binary average).
        f1: F1 score on the test set (binary average).
        fit_time_seconds: Wall-clock time to fit the model.
        predict_time_seconds: Wall-clock time to predict on the test set.
        depth: Depth of the fitted tree.
        n_leaves: Number of leaves in the fitted tree.
    """

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    fit_time_seconds: float
    predict_time_seconds: float
    depth: int
    n_leaves: int


def _time_fn(fn: Callable[[], object]) -> tuple[object, float]:
    """Time execution of a zero-argument callable.

    Args:
        fn: Callable to execute and time.

    Returns:
        Tuple of (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _benchmark_treeforge(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_depth: int,
) -> BenchmarkResult:
    """Benchmark the TreeForge classifier.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        max_depth: Maximum tree depth.

    Returns:
        BenchmarkResult for TreeForge.
    """
    clf = TreeForgeClassifier(max_depth=max_depth)
    _, fit_time = _time_fn(lambda: clf.fit(X_train, y_train))
    y_pred, predict_time = _time_fn(lambda: clf.predict(X_test))

    return BenchmarkResult(
        name="TreeForge",
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="binary"),
        recall=recall_score(y_test, y_pred, average="binary"),
        f1=f1_score(y_test, y_pred, average="binary"),
        fit_time_seconds=fit_time,
        predict_time_seconds=predict_time,
        depth=clf.tree_.get_depth(),
        n_leaves=clf.tree_.get_n_leaves(),
    )


def _benchmark_sklearn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_depth: int,
) -> BenchmarkResult:
    """Benchmark sklearn's DecisionTreeClassifier.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        max_depth: Maximum tree depth.

    Returns:
        BenchmarkResult for sklearn.
    """
    clf = SklearnDecisionTreeClassifier(max_depth=max_depth, random_state=42)
    _, fit_time = _time_fn(lambda: clf.fit(X_train, y_train))
    y_pred, predict_time = _time_fn(lambda: clf.predict(X_test))

    return BenchmarkResult(
        name="sklearn",
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="binary"),
        recall=recall_score(y_test, y_pred, average="binary"),
        f1=f1_score(y_test, y_pred, average="binary"),
        fit_time_seconds=fit_time,
        predict_time_seconds=predict_time,
        depth=clf.get_depth(),
        n_leaves=clf.get_n_leaves(),
    )


def run_benchmark(
    max_depth: int = 10, test_size: float = 0.2, random_state: int = 42
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Run TreeForge vs sklearn benchmark on the MAGIC dataset.

    Args:
        max_depth: Maximum tree depth applied to both models.
        test_size: Proportion of data reserved for testing.
        random_state: Seed for reproducible splitting.

    Returns:
        Tuple of (treeforge_result, sklearn_result).
    """
    df = load_and_process(persist=False)
    X = df.drop(columns=[TARGET_COLUMN]).to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    treeforge_result = _benchmark_treeforge(
        X_train, X_test, y_train, y_test, max_depth
    )
    sklearn_result = _benchmark_sklearn(X_train, X_test, y_train, y_test, max_depth)

    return treeforge_result, sklearn_result


def format_comparison_table(
    treeforge_result: BenchmarkResult, sklearn_result: BenchmarkResult
) -> str:
    """Format a human-readable side-by-side comparison table.

    Args:
        treeforge_result: Benchmark result for TreeForge.
        sklearn_result: Benchmark result for sklearn.

    Returns:
        Formatted multi-line string table.
    """
    rows = [
        ("Metric", treeforge_result.name, sklearn_result.name),
        ("Accuracy", f"{treeforge_result.accuracy:.4f}", f"{sklearn_result.accuracy:.4f}"),
        ("Precision", f"{treeforge_result.precision:.4f}", f"{sklearn_result.precision:.4f}"),
        ("Recall", f"{treeforge_result.recall:.4f}", f"{sklearn_result.recall:.4f}"),
        ("F1 Score", f"{treeforge_result.f1:.4f}", f"{sklearn_result.f1:.4f}"),
        (
            "Fit Time (s)",
            f"{treeforge_result.fit_time_seconds:.4f}",
            f"{sklearn_result.fit_time_seconds:.4f}",
        ),
        (
            "Predict Time (s)",
            f"{treeforge_result.predict_time_seconds:.4f}",
            f"{sklearn_result.predict_time_seconds:.4f}",
        ),
        ("Tree Depth", str(treeforge_result.depth), str(sklearn_result.depth)),
        ("Leaves", str(treeforge_result.n_leaves), str(sklearn_result.n_leaves)),
    ]

    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = []
    for row in rows:
        line = " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if row == rows[0]:
            lines.append("-" * len(line))

    return "\n".join(lines)


if __name__ == "__main__":
    tf_result, sk_result = run_benchmark()
    print(format_comparison_table(tf_result, sk_result))