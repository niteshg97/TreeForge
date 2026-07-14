"""Unit tests for src.benchmark.compare module."""

import numpy as np

from src.benchmark.compare import (BenchmarkResult, _benchmark_sklearn,
                                   _benchmark_treeforge,
                                   format_comparison_table)


def _separable_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.array([[1.0], [2.0], [8.0], [9.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5], [8.5]])
    y_test = np.array([0, 1])
    return X_train, X_test, y_train, y_test


def test_benchmark_treeforge_returns_valid_result() -> None:
    X_train, X_test, y_train, y_test = _separable_dataset()
    result = _benchmark_treeforge(X_train, X_test, y_train, y_test, max_depth=3)
    assert isinstance(result, BenchmarkResult)
    assert result.name == "TreeForge"
    assert 0.0 <= result.accuracy <= 1.0
    assert result.fit_time_seconds >= 0.0
    assert result.depth >= 0


def test_benchmark_sklearn_returns_valid_result() -> None:
    X_train, X_test, y_train, y_test = _separable_dataset()
    result = _benchmark_sklearn(X_train, X_test, y_train, y_test, max_depth=3)
    assert isinstance(result, BenchmarkResult)
    assert result.name == "sklearn"
    assert 0.0 <= result.accuracy <= 1.0
    assert result.fit_time_seconds >= 0.0
    assert result.depth >= 0


def test_both_models_perfect_on_separable_data() -> None:
    X_train, X_test, y_train, y_test = _separable_dataset()
    tf_result = _benchmark_treeforge(X_train, X_test, y_train, y_test, max_depth=3)
    sk_result = _benchmark_sklearn(X_train, X_test, y_train, y_test, max_depth=3)
    assert tf_result.accuracy == 1.0
    assert sk_result.accuracy == 1.0


def test_format_comparison_table_contains_headers() -> None:
    X_train, X_test, y_train, y_test = _separable_dataset()
    tf_result = _benchmark_treeforge(X_train, X_test, y_train, y_test, max_depth=3)
    sk_result = _benchmark_sklearn(X_train, X_test, y_train, y_test, max_depth=3)
    table = format_comparison_table(tf_result, sk_result)
    assert "Accuracy" in table
    assert "TreeForge" in table
    assert "sklearn" in table
    assert "Tree Depth" in table


def test_format_comparison_table_row_count() -> None:
    X_train, X_test, y_train, y_test = _separable_dataset()
    tf_result = _benchmark_treeforge(X_train, X_test, y_train, y_test, max_depth=3)
    sk_result = _benchmark_sklearn(X_train, X_test, y_train, y_test, max_depth=3)
    table = format_comparison_table(tf_result, sk_result)
    # header + separator + 8 metric rows
    assert len(table.splitlines()) == 10
