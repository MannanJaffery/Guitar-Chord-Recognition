"""Dataset loading utilities.

The upstream repo (j-asra-j/guitar-chords) contains filenames with ':' which
cannot be written on Windows, so we load the Hugging Face auto-converted
Parquet branch instead. It is byte-identical in content and works everywhere.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from datasets import DatasetDict, load_dataset

PARQUET_BASE = "hf://datasets/{repo}@refs%2Fconvert%2Fparquet/default/"
SPLIT_FILES = {
    "train": ["train/0000.parquet", "train/0001.parquet"],
    "validation": ["validation/0000.parquet"],
    "test": ["test/0000.parquet"],
}

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "configs" / "dataset.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["dataset"]


def load_guitar_chords(repo: str = "j-asra-j/guitar-chords", cache_dir: str | Path | None = None) -> DatasetDict:
    """Load all splits as a DatasetDict with columns: image (PIL), label (ClassLabel)."""
    base = PARQUET_BASE.format(repo=repo)
    data_files = {split: [base + f for f in files] for split, files in SPLIT_FILES.items()}
    return load_dataset("parquet", data_files=data_files, cache_dir=str(cache_dir) if cache_dir else None)
