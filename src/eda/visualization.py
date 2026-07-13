"""Visualization utilities for exploratory data analysis."""

from typing import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_class_distribution(
    df: pd.DataFrame, target_col: str
) -> matplotlib.figure.Figure:
    """Plot bar chart of class distribution.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target_col].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> matplotlib.figure.Figure:
    """Plot histograms for a list of numeric feature columns.

    Args:
        df: Input DataFrame.
        feature_cols: Feature column names to plot.

    Returns:
        Matplotlib Figure object with subplots.
    """
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=30, color="steelblue", edgecolor="black")
        axes[idx].set_title(col)

    for idx in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Plot heatmap of a correlation matrix.

    Args:
        corr_matrix: Precomputed correlation matrix.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig