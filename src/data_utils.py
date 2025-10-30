from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd

# Canonical feature/target names we will use throughout
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "species"


def _lower_strip(cols):
    return {c: c.lower().strip() for c in cols}


def standardize_iris_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names from various Iris sources to:
      sepal_length, sepal_width, petal_length, petal_width, species
    Drops common ID columns if found.
    """
    # Normalize case/whitespace
    df = df.rename(columns=_lower_strip(df.columns))

    # Map common Kaggle/Sklearn names -> canonical names
    rename_map = {
        "sepallengthcm": "sepal_length",
        "sepalwidthcm": "sepal_width",
        "petallengthcm": "petal_length",
        "petalwidthcm": "petal_width",
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "variety": "species",  # some iris variants use 'variety'
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # Drop ID-like columns if present
    for id_col in ["id", "index", "unnamed: 0"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    return df


def load_iris_csv(path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load Iris CSV, standardize columns, return X, y, and the cleaned dataframe.
    If 'path' is a directory, tries common filenames like Iris.csv.
    """
    p = Path(path)
    if p.is_dir():
        for name in ["Iris.csv", "iris.csv", "IRIS.csv"]:
            candidate = p / name
            if candidate.exists():
                p = candidate
                break

    if not p.exists():
        raise FileNotFoundError(f"Could not find dataset at: {p}")

    df = pd.read_csv(p)
    df = standardize_iris_columns(df)

    missing = [c for c in IRIS_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    X = df[IRIS_FEATURES].copy()
    y = df[TARGET_COL].astype(str).copy()  # keep labels as strings
    return X, y, df


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )