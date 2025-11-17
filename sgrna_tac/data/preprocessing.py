"""Dataset preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PKD_REQUIRED_COLUMNS = {
    "WTSequence (WildType)",
    "MutatedSequence",
    "Day21-ETP-binarized",
}


def load_and_preprocess_pkd(file_path: str | Path) -> pd.DataFrame:
    """Load the PKD dataset and normalize column names."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PKD dataset not found: {path}")

    df = pd.read_csv(path)
    missing = PKD_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"PKD dataset missing columns: {sorted(missing)}")

    df = df[list(PKD_REQUIRED_COLUMNS)]
    df = df.rename(columns={"Day21-ETP-binarized": "Label"})
    df = df.dropna().reset_index(drop=True)
    return df


__all__ = ["load_and_preprocess_pkd"]

