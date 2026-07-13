"""Unit tests for src.eda.visualization module."""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.eda.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_distributions,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "class": [1, 0, 1, 0, 1],
        }
    )


def test_plot_class_distribution_returns_figure(sample_df: pd.DataFrame) -> None:
    fig = plot_class_distribution(sample_df, "class")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_distributions_returns_figure(sample_df: pd.DataFrame) -> None:
    fig = plot_feature_distributions(sample_df, ["feat_a", "feat_b"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_heatmap_returns_figure(sample_df: pd.DataFrame) -> None:
    corr = sample_df[["feat_a", "feat_b"]].corr()
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)