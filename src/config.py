"""Project-wide configuration constants for TreeForge."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "magic04.data"
PROCESSED_DATA_PATH: Path = (
    PROJECT_ROOT / "data" / "processed" / "magic04_processed.csv"
)

COLUMN_NAMES: list[str] = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "class",
]

TARGET_COLUMN: str = "class"
CLASS_MAPPING: dict[str, int] = {"g": 1, "h": 0}
