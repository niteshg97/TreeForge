"""Unit tests for src.eda.summary module."""

import pandas as pd
import pytest

from src.eda.summary import (
    get_class_distribution,
    get_correlation_matrix,
    get_summary_statistics,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0],
            "feat_b": [4.0, 3.0, 2.0, 1.0],
            "class": [1, 0, 1, 0],
        }
    )


def test_get_summary_statistics_shape(sample_df: pd.DataFrame) -> None:
    stats = get_summary_statistics(sample_df)
    assert "mean" in stats.columns
    assert "feat_a" in stats.index


def test_get_class_distribution_values(sample_df: pd.DataFrame) -> None:
    dist = get_class_distribution(sample_df, "class")
    assert dist.loc[1, "count"] == 2
    assert dist.loc[0, "proportion"] == 0.5


def test_get_class_distribution_missing_column_raises(
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(KeyError):
        get_class_distribution(sample_df, "missing")


def test_get_correlation_matrix_symmetric(sample_df: pd.DataFrame) -> None:
    corr = get_correlation_matrix(sample_df[["feat_a", "feat_b"]])
    assert corr.loc["feat_a", "feat_b"] == corr.loc["feat_b", "feat_a"]