from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import DATA_DIR


@st.cache_resource(ttl=86400, max_entries=20)
def _load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return _optimize_dataframe_memory(df)
    df = pd.read_csv(path)
    return _optimize_dataframe_memory(df)


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory footprint by optimizing dtypes"""
    if df.empty:
        return df

    # Convert object columns to category where appropriate (50%+ savings)
    for col in df.select_dtypes(include=["object"]).columns:
        # Skip ID columns
        if col.endswith("_mlbid") or col.endswith("_id"):
            continue
        num_unique = df[col].nunique()
        num_total = len(df[col])
        # If less than 50% unique values, use category
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype("category")

    # Downcast numeric types (30-50% savings)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if path.suffix == ".csv":
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            path = parquet_path
    if not path.exists():
        return pd.DataFrame()
    return _load_csv_cached(str(path), path.stat().st_mtime)


def load_damage_df() -> pd.DataFrame:
    # Pick the most recent full multi-season file (damage_pos_2015_YYYY.parquet)
    candidates = sorted(DATA_DIR.glob("damage_pos_2015_[0-9][0-9][0-9][0-9].parquet"))
    if candidates:
        path = candidates[-1]
        return _load_csv_cached(str(path), path.stat().st_mtime)
    candidates = sorted(DATA_DIR.glob("damage_pos_2015_[0-9][0-9][0-9][0-9].csv"))
    if candidates:
        path = candidates[-1]
        return _load_csv_cached(str(path), path.stat().st_mtime)
    return pd.DataFrame()
