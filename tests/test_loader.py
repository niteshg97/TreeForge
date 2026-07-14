"""Unit tests for src.data.loader module."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import COLUMN_NAMES
from src.data.loader import (encode_target, load_and_process, load_raw_data,
                             save_processed_data)


@pytest.fixture
def sample_raw_csv(tmp_path: Path) -> Path:
    """Create a small sample raw data file."""
    content = (
        "1.1,2.2,3.3,0.1,0.2,10.0,5.0,6.0,20.0,100.0,g\n"
        "2.1,3.2,4.3,0.3,0.4,15.0,7.0,8.0,25.0,150.0,h\n"
    )
    path = tmp_path / "sample.data"
    path.write_text(content)
    return path


def test_load_raw_data_returns_correct_columns(sample_raw_csv: Path) -> None:
    df = load_raw_data(sample_raw_csv)
    assert list(df.columns) == COLUMN_NAMES
    assert len(df) == 2


def test_load_raw_data_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("nonexistent.data"))


def test_encode_target_maps_labels(sample_raw_csv: Path) -> None:
    df = load_raw_data(sample_raw_csv)
    encoded = encode_target(df)
    assert set(encoded["class"].unique()) == {0, 1}


def test_encode_target_invalid_label_raises(sample_raw_csv: Path) -> None:
    df = load_raw_data(sample_raw_csv)
    df.loc[0, "class"] = "x"
    with pytest.raises(ValueError):
        encode_target(df)


def test_save_processed_data_writes_file(sample_raw_csv: Path, tmp_path: Path) -> None:
    df = encode_target(load_raw_data(sample_raw_csv))
    output_path = tmp_path / "processed" / "out.csv"
    save_processed_data(df, output_path)
    assert output_path.exists()
    loaded = pd.read_csv(output_path)
    assert len(loaded) == len(df)


def test_load_and_process_pipeline(sample_raw_csv: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "processed.csv"
    df = load_and_process(sample_raw_csv, output_path, persist=True)
    assert output_path.exists()
    assert df["class"].isin([0, 1]).all()
