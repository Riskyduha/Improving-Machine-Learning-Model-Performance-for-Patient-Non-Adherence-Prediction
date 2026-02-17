"""Project-wide configuration helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = BASE_DIR / "results"

# Global experiment configuration
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
TARGET_COLUMN: str = "future_non_adherence"


def ensure_directories(extra_paths: Iterable[Path] | None = None) -> None:
    """Create core folders (and optional extras) when they do not exist."""
    base_targets = {DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR}
    if extra_paths:
        base_targets.update(extra_paths)
    for path in base_targets:
        path.mkdir(parents=True, exist_ok=True)
