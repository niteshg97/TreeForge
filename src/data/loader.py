"""Data loading and preprocessing utilities for the MAGIC dataset."""

from pathlib import Path

import pandas as pd

from src.config import (CLASS_MAPPING, COLUMN_NAMES, PROCESSED_DATA_PATH,
                        RAW_DATA_PATH, TARGET_COLUMN)


def load_raw_data(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw MAGIC dataset from disk.

    Args:
        raw_path: Path to the raw .data file.

    Returns:
        DataFrame with named columns.

    Raises:
        FileNotFoundError: If the raw file does not exist.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    return pd.read_csv(raw_path, header=None, names=COLUMN_NAMES)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target class labels as integers.

    Args:
        df: DataFrame containing the raw target column.

    Returns:
        DataFrame with encoded target column.

    Raises:
        ValueError: If an unmapped class label is found.
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(CLASS_MAPPING)
    if df[TARGET_COLUMN].isnull().any():
        raise ValueError("Unknown class label encountered during encoding.")
    return df


def save_processed_data(
    df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH
) -> None:
    """Persist processed DataFrame to disk as CSV.

    Args:
        df: Processed DataFrame to save.
        output_path: Destination path for the processed CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_and_process(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    persist: bool = True,
) -> pd.DataFrame:
    """Run full pipeline: load raw data, encode target, optionally persist.

    Args:
        raw_path: Path to raw data file.
        output_path: Path to save processed data.
        persist: Whether to write processed data to disk.

    Returns:
        Fully processed DataFrame.
    """
    df = load_raw_data(raw_path)
    df = encode_target(df)
    if persist:
        save_processed_data(df, output_path)
    return df
