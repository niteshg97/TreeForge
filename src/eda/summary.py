"""Statistical summary utilities for exploratory data analysis."""

import pandas as pd


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for all numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame of descriptive statistics (count, mean, std, etc.).
    """
    return df.describe().transpose()


def get_class_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute class counts and proportions for the target column.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        DataFrame with 'count' and 'proportion' per class.

    Raises:
        KeyError: If target_col is not present in df.
    """
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in DataFrame.")
    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)
    return pd.DataFrame({"count": counts, "proportion": proportions})


def get_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute pairwise correlation matrix for numeric columns.

    Args:
        df: Input DataFrame.
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        Correlation matrix as a DataFrame.
    """
    return df.corr(method=method)
