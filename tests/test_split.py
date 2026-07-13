"""Unit tests for src.model_selection.split module."""

import numpy as np
import pytest

from src.model_selection.split import train_test_split


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)
    return X, y


def test_split_sizes(sample_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    assert X_test.shape[0] == 3
    assert X_train.shape[0] == 7
    assert y_train.shape[0] == 7
    assert y_test.shape[0] == 3


def test_split_reproducible_with_same_seed(
    sample_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_data
    result_a = train_test_split(X, y, test_size=0.2, random_state=1)
    result_b = train_test_split(X, y, test_size=0.2, random_state=1)
    for arr_a, arr_b in zip(result_a, result_b):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_split_different_seeds_differ(
    sample_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_data
    X_train_1, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train_2, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=2)
    assert not np.array_equal(X_train_1, X_train_2)


def test_split_invalid_test_size_raises(
    sample_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_data
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=0.0)


def test_split_no_overlap_between_train_test(
    sample_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_data
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=5)
    train_rows = {tuple(row) for row in X_train}
    test_rows = {tuple(row) for row in X_test}
    assert train_rows.isdisjoint(test_rows)