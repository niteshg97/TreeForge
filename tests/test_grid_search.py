"""Unit tests for src.model_selection.grid_search module."""

import numpy as np

from src.model_selection.grid_search import GridSearchCV, GridSearchResult


def _separable_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([np.zeros((10, 1)), np.ones((10, 1))]).astype(float) * 10
    y = np.array([0] * 10 + [1] * 10)
    return X, y


def test_grid_search_returns_result_type() -> None:
    X, y = _separable_dataset()
    grid = GridSearchCV(param_grid={"max_depth": [1, 3]}, n_splits=4)
    result = grid.fit(X, y)
    assert isinstance(result, GridSearchResult)


def test_grid_search_evaluates_all_combinations() -> None:
    X, y = _separable_dataset()
    grid = GridSearchCV(
        param_grid={"max_depth": [1, 2], "min_samples_split": [2, 4]}, n_splits=4
    )
    result = grid.fit(X, y)
    assert len(result.all_results) == 4


def test_grid_search_selects_best_score() -> None:
    X, y = _separable_dataset()
    grid = GridSearchCV(param_grid={"max_depth": [1, 5]}, n_splits=4)
    result = grid.fit(X, y)
    best_manual = max(result.all_results, key=lambda r: r["mean_score"])
    assert result.best_params == best_manual["params"]
    assert result.best_score == best_manual["mean_score"]


def test_grid_search_best_score_within_valid_range() -> None:
    X, y = _separable_dataset()
    grid = GridSearchCV(param_grid={"max_depth": [1, 3]}, n_splits=4)
    result = grid.fit(X, y)
    assert 0.0 <= result.best_score <= 1.0
