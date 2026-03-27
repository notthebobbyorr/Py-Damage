from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from app.auth import _is_user_subscribed
from app.config import (
    GAME_TYPE_GROUP_NOTE,
    GAME_TYPE_GROUP_OPTIONS,
    POSITION_COUNT_THRESHOLD,
    POSITION_FILTER_COLS,
    PREVIEW_ROWS,
)


def season_options(df: pd.DataFrame, column: str = "season") -> list:
    if df.empty or column not in df.columns:
        return ["All"]
    values = pd.Series(df[column].dropna().unique())
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        sorted_vals = values.loc[numeric.sort_values(ascending=False).index].tolist()
    else:
        sorted_vals = values.sort_values(ascending=False).tolist()
    return ["All"] + sorted_vals


def filter_by_values(df: pd.DataFrame, column: str, values) -> pd.DataFrame:
    if df.empty:
        return df
    if values is None:
        return df
    if isinstance(values, (str, bytes)):
        if values == "All":
            return df
        return df[df[column] == values]
    if not isinstance(values, (list, tuple, set, pd.Index, np.ndarray, pd.Series)):
        return df[df[column] == values]
    values_list = list(values)
    if not values_list or "All" in values_list:
        return df
    return df[df[column].isin(values_list)]


def _split_team_tokens(value: str) -> list[str]:
    tokens = [token.strip() for token in re.split(r"[|,/]", value)]
    return [token for token in tokens if token]


def team_options(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return ["All"]
    tokens: set[str] = set()
    for value in df[column].dropna().astype(str):
        tokens.update(_split_team_tokens(value))
    return ["All"] + sorted(tokens)


def filter_by_team_token(df: pd.DataFrame, column: str, team: str) -> pd.DataFrame:
    if df.empty or team == "All":
        return df
    if column not in df.columns:
        return df
    mask = df[column].astype(str).apply(lambda v: team in _split_team_tokens(v))
    return df[mask]


def position_options(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["All"]
    options = [
        pos
        for pos in POSITION_FILTER_COLS
        if pos in df.columns or f"is_{pos}" in df.columns
    ]
    return ["All"] + options if options else ["All"]


def filter_by_positions(
    df: pd.DataFrame,
    positions,
    min_count: int = POSITION_COUNT_THRESHOLD,
) -> pd.DataFrame:
    if df.empty or positions is None:
        return df
    if isinstance(positions, (str, bytes)):
        positions = [positions]
    if not isinstance(positions, (list, tuple, set, pd.Index, np.ndarray, pd.Series)):
        positions = [positions]
    selected = [pos for pos in positions if pos != "All"]
    if not selected:
        return df
    mask = pd.Series(False, index=df.index)
    for pos in selected:
        binary_col = f"is_{pos}"
        if binary_col in df.columns:
            values = pd.to_numeric(df[binary_col], errors="coerce").fillna(0)
            mask |= values >= 1
            continue
        if pos not in df.columns:
            continue
        values = pd.to_numeric(df[pos], errors="coerce").fillna(0)
        threshold = 1 if values.max(skipna=True) <= 1 else min_count
        mask |= values >= threshold
    if not mask.any():
        return df.iloc[0:0]
    return df[mask]


def player_id_options(
    df: pd.DataFrame, id_col: str, name_col: str
) -> tuple[list, dict]:
    if df.empty or id_col not in df.columns:
        return ["All"], {}
    options_df = (
        df[[id_col, name_col]].copy() if name_col in df.columns else df[[id_col]].copy()
    )
    options_df[id_col] = pd.to_numeric(options_df[id_col], errors="coerce")
    options_df = options_df.dropna(subset=[id_col])
    if name_col in options_df.columns:
        options_df[name_col] = options_df[name_col].astype(str)
    options_df = options_df.drop_duplicates(subset=[id_col])
    if name_col in options_df.columns:
        options_df = options_df.sort_values(by=[name_col, id_col])
        name_map = dict(zip(options_df[id_col], options_df[name_col]))
    else:
        options_df = options_df.sort_values(by=[id_col])
        name_map = {}
    ids = options_df[id_col].tolist()
    return ["All"] + ids, name_map


def numeric_filter(df: pd.DataFrame, column: str, min_value: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df[column] >= min_value]


def game_type_group_options(df: pd.DataFrame) -> list[str]:
    if df.empty or "game_type_group" not in df.columns:
        return GAME_TYPE_GROUP_OPTIONS
    available = df["game_type_group"].dropna().unique().tolist()
    return [g for g in GAME_TYPE_GROUP_OPTIONS if g in available] or GAME_TYPE_GROUP_OPTIONS


def filter_by_game_type_group(df: pd.DataFrame, game_type_group: str) -> pd.DataFrame:
    if df.empty or "game_type_group" not in df.columns:
        # Old data (pre-game_type_group) — treat as Regular Season
        if game_type_group == "Regular Season":
            return df
        return df.iloc[0:0]  # empty frame for other types
    if game_type_group == "Regular Season":
        return df[(df["game_type_group"] == "Regular Season") | df["game_type_group"].isna()]
    return df[df["game_type_group"] == game_type_group]


def pitcher_workload_filter(
    df: pd.DataFrame, filter_type: str, min_value: float
) -> pd.DataFrame:
    if df.empty:
        return df
    metric = filter_type if filter_type in {"IP", "TBF", "GS"} else "TBF"
    if metric not in df.columns:
        return df
    return numeric_filter(df, metric, min_value)


def download_button(df: pd.DataFrame, label: str, key: str) -> None:
    if df.empty:
        return
    if not _is_user_subscribed():
        st.info("Subscribe to download the full dataset.")
        return
    csv = df.to_csv(index=False)
    st.download_button(label, data=csv, file_name=f"{label}.csv", key=key)


def apply_column_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    with st.expander("Column filters", expanded=False):
        filtered = df
        for col in df.columns:
            if col.startswith("__"):
                continue
            col_key = f"{key_prefix}_{col}"
            if pd.api.types.is_numeric_dtype(df[col]):
                op = st.selectbox(
                    f"{col} filter",
                    options=["(no filter)", "=", "<", "<=", ">", ">=", "between"],
                    key=f"{col_key}_op",
                )
                if op == "(no filter)":
                    continue
                if op == "between":
                    low = st.number_input(f"{col} min", key=f"{col_key}_min", value=0.0)
                    high = st.number_input(
                        f"{col} max", key=f"{col_key}_max", value=0.0
                    )
                    filtered = filtered[
                        (filtered[col] >= low) & (filtered[col] <= high)
                    ]
                else:
                    value = st.number_input(
                        f"{col} value", key=f"{col_key}_val", value=0.0
                    )
                    if op == "=":
                        filtered = filtered[filtered[col] == value]
                    elif op == "<":
                        filtered = filtered[filtered[col] < value]
                    elif op == "<=":
                        filtered = filtered[filtered[col] <= value]
                    elif op == ">":
                        filtered = filtered[filtered[col] > value]
                    elif op == ">=":
                        filtered = filtered[filtered[col] >= value]
            else:
                op = st.selectbox(
                    f"{col} filter",
                    options=["(no filter)", "=", "contains"],
                    key=f"{col_key}_op",
                )
                if op == "(no filter)":
                    continue
                value = st.text_input(f"{col} value", key=f"{col_key}_val", value="")
                if value:
                    if op == "=":
                        filtered = filtered[filtered[col] == value]
                    else:
                        filtered = filtered[
                            filtered[col]
                            .astype(str)
                            .str.contains(value, case=False, na=False)
                        ]
        return filtered
