"""Unit tests for src.tree.classifier module."""

import numpy as np
import pytest

from src.tree.classifier import DecisionTreeClassifier, NotFittedError


@pytest.fixture
def separable_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_fit_returns_self(separable_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = separable_data
    clf = DecisionTreeClassifier()
    result = clf.fit(X, y)
    assert result is clf


def test_predict_perfect_separation(
    separable_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = separable_data
    clf = DecisionTreeClassifier().fit(X, y)
    preds = clf.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_predict_proba_shape(
    separable_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = separable_data
    clf = DecisionTreeClassifier().fit(X, y)
    probas = clf.predict_proba(X)
    assert probas.shape == (4, 2)
    np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(4))


def test_predict_before_fit_raises() -> None:
    clf = DecisionTreeClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(np.array([[1.0]]))


def test_predict_proba_before_fit_raises() -> None:
    clf = DecisionTreeClassifier()
    with pytest.raises(NotFittedError):
        clf.predict_proba(np.array([[1.0]]))


def test_classes_and_n_classes_attributes(
    separable_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = separable_data
    clf = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(clf.classes_, [0, 1])
    assert clf.n_classes_ == 2


def test_fit_with_string_labels() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array(["cat", "cat", "dog", "dog"])
    clf = DecisionTreeClassifier().fit(X, y)
    preds = clf.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_fit_multi_class() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.n_classes_ == 3
    probas = clf.predict_proba(X)
    assert probas.shape == (6, 3)


def test_invalid_X_dimension_raises() -> None:
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))


def test_mismatched_X_y_length_raises() -> None:
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.array([[1.0], [2.0]]), np.array([0, 1, 1]))
