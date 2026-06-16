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


_CUSTOM_COL_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b.replace(0, np.nan),
}


def _apply_custom_columns(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Append session-defined custom columns to df (computed left-to-right)."""
    state_key = f"{key_prefix}_custom_cols"
    defs = st.session_state.get(state_key, [])
    if not defs:
        return df
    out = df
    for spec in defs:
        name = spec.get("name")
        base = spec.get("base_cols", [])
        op = spec.get("op")
        if not name or len(base) < 2 or op not in _CUSTOM_COL_OPS:
            continue
        if not all(col in out.columns for col in base):
            continue
        try:
            series = pd.to_numeric(out[base[0]], errors="coerce")
            for other in base[1:]:
                other_series = pd.to_numeric(out[other], errors="coerce")
                series = _CUSTOM_COL_OPS[op](series, other_series)
            series = series.replace([np.inf, -np.inf], np.nan)
            out = out.copy()
            out[name] = series
        except Exception:
            continue
    return out


def _render_custom_column_builder(df: pd.DataFrame, key_prefix: str) -> None:
    state_key = f"{key_prefix}_custom_cols"
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    st.caption("**Custom columns** — build a derived column from existing numeric columns.")
    numeric_cols = [
        col for col in df.columns
        if not col.startswith("__") and pd.api.types.is_numeric_dtype(df[col])
    ]
    existing = st.session_state[state_key]
    existing_names = {spec["name"] for spec in existing}
    candidate_cols = [c for c in numeric_cols if c not in existing_names]

    if not candidate_cols or len(candidate_cols) < 2:
        st.caption("Need at least 2 numeric columns to build a custom column.")
    else:
        builder_col1, builder_col2 = st.columns([3, 1])
        with builder_col1:
            base = st.multiselect(
                "Base columns (applied left-to-right)",
                options=candidate_cols,
                key=f"{key_prefix}_custom_base",
                max_selections=4,
            )
        with builder_col2:
            op = st.selectbox(
                "Operator",
                options=list(_CUSTOM_COL_OPS.keys()),
                key=f"{key_prefix}_custom_op",
            )
        name = st.text_input(
            "Column name",
            key=f"{key_prefix}_custom_name",
            placeholder="e.g. HR rate",
        )
        if st.button("Add column", key=f"{key_prefix}_custom_add"):
            cleaned_name = (name or "").strip()
            if len(base) < 2:
                st.warning("Select at least 2 base columns.")
            elif not cleaned_name:
                st.warning("Give the column a name.")
            elif cleaned_name in df.columns or cleaned_name in existing_names:
                st.warning(f"A column named '{cleaned_name}' already exists.")
            else:
                st.session_state[state_key].append(
                    {"name": cleaned_name, "base_cols": list(base), "op": op}
                )
                st.rerun()

    if existing:
        st.caption("Active custom columns:")
        for idx, spec in enumerate(existing):
            row1, row2 = st.columns([5, 1])
            with row1:
                expr = f" {spec['op']} ".join(spec["base_cols"])
                st.write(f"**{spec['name']}** = `{expr}`")
            with row2:
                if st.button("Remove", key=f"{key_prefix}_custom_rm_{idx}"):
                    st.session_state[state_key].pop(idx)
                    st.rerun()


def apply_column_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = _apply_custom_columns(df, key_prefix)
    with st.expander("Column filters", expanded=False):
        _render_custom_column_builder(df, key_prefix)
        st.markdown("---")
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
