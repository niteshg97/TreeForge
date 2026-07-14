"""Unit tests for src.eda.visualization module."""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.eda.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_distributions,
)

matplotlib.use("Agg")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small sample dataframe."""
    return pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "class": [1, 0, 1, 0, 1],
        }
    )


def test_plot_class_distribution_returns_figure(
    sample_df: pd.DataFrame,
) -> None:
    """plot_class_distribution should return a matplotlib Figure."""
    fig = plot_class_distribution(sample_df, "class")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_distributions_returns_figure(
    sample_df: pd.DataFrame,
) -> None:
    """plot_feature_distributions should return a matplotlib Figure."""
    fig = plot_feature_distributions(
        sample_df,
        ["feat_a", "feat_b"],
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_heatmap_returns_figure(
    sample_df: pd.DataFrame,
) -> None:
    """plot_correlation_heatmap should return a matplotlib Figure."""
    corr = sample_df[["feat_a", "feat_b"]].corr()
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
