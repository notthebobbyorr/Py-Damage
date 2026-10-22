from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import DATA_DIR
from app.swing_rates import add_swing_rates

_NAME_COLS = {"hitter_name", "name", "pitcher_name", "runner_name"}


def _to_ascii(val: object) -> object:
    if pd.isna(val):
        return val
    return unicodedata.normalize("NFKD", str(val)).encode("ascii", "ignore").decode("ascii")


@st.cache_resource(ttl=86400, max_entries=20)
def _load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        df = add_swing_rates(_read_parquet(path))
        return _optimize_dataframe_memory(df)
    df = add_swing_rates(pd.read_csv(path))
    return _optimize_dataframe_memory(df)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Accept Polars dictionary encodings as well as pandas-written parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for index, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type) and pa.types.is_unsigned_integer(field.type.index_type):
            signed = pa.dictionary(pa.int64(), field.type.value_type, ordered=field.type.ordered)
            table = table.set_column(index, field.name, table.column(index).cast(signed))
    return table.to_pandas()


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory footprint by optimizing dtypes"""
    if df.empty:
        return df

    # Normalize player name columns to ASCII (strip diacritics)
    for col in df.columns:
        if col in _NAME_COLS:
            df[col] = df[col].map(_to_ascii)

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
        if parquet_path.exists() and parquet_path.stat().st_size > 0:
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
