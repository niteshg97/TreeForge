"""Unit tests for src.evaluation.metrics module."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def test_accuracy_score_perfect() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert accuracy_score(y_true, y_pred) == 1.0


def test_accuracy_score_partial() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)


def test_confusion_matrix_binary() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    expected = np.array([[1, 1], [0, 2]])
    np.testing.assert_array_equal(cm, expected)


def test_precision_score_binary() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert precision_score(y_true, y_pred, average="binary") == pytest.approx(2 / 3)


def test_recall_score_binary() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert recall_score(y_true, y_pred, average="binary") == pytest.approx(1.0)


def test_f1_score_binary() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    precision = 2 / 3
    recall = 1.0
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert f1_score(y_true, y_pred, average="binary") == pytest.approx(expected_f1)


def test_macro_average_multi_class() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 0, 2])
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_invalid_average_raises() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError):
        precision_score(y_true, y_pred, average="invalid")