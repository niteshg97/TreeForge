"""Exhaustive grid search with cross-validation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.evaluation.metrics import accuracy_score
from src.model_selection.cross_validation import ScoringFunc, cross_val_score


@dataclass
class GridSearchResult:
    """Holds the outcome of a grid search run.

    Attributes:
        best_params: Hyperparameter combination with the highest mean score.
        best_score: Mean CV score achieved by best_params.
        all_results: List of dicts with 'params' and 'mean_score' per combo.
    """

    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]] = field(default_factory=list)


class GridSearchCV:
    """Exhaustive search over a hyperparameter grid using cross-validation.

    Attributes:
        param_grid: Mapping of hyperparameter name to list of candidate values.
        n_splits: Number of cross-validation folds.
        scoring: Scoring function taking (y_true, y_pred) -> float.
        random_state: Seed for reproducible fold generation.
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        n_splits: int = 5,
        scoring: ScoringFunc = accuracy_score,
        random_state: int = 42,
    ) -> None:
        """Initialize the grid search.

        Args:
            param_grid: Mapping of hyperparameter name to candidate values.
            n_splits: Number of cross-validation folds.
            scoring: Scoring function taking (y_true, y_pred) -> float.
            random_state: Seed for reproducible fold generation.
        """
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state

    def _generate_param_combinations(self) -> list[dict[str, Any]]:
        """Generate all combinations of hyperparameters from the grid.

        Returns:
            List of parameter dictionaries, one per combination.
        """
        keys = list(self.param_grid.keys())
        value_combinations = itertools.product(*self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in value_combinations]

    def fit(self, X: np.ndarray, y: np.ndarray) -> GridSearchResult:
        """Run exhaustive grid search with cross-validation.

        Args:
            X: 2D feature matrix (n_samples, n_features).
            y: 1D label array.

        Returns:
            GridSearchResult with best params, best score, and all results.
        """
        all_results: list[dict[str, Any]] = []
        best_score = -np.inf
        best_params: dict[str, Any] = {}

        for params in self._generate_param_combinations():
            scores = cross_val_score(
                X,
                y,
                n_splits=self.n_splits,
                scoring=self.scoring,
                random_state=self.random_state,
                **params,
            )
            mean_score = float(np.mean(scores))
            all_results.append({"params": params, "mean_score": mean_score})

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        return GridSearchResult(
            best_params=best_params, best_score=best_score, all_results=all_results
        )
