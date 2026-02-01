from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import colors
import plotly.express as px
from st_paywall import add_auth

DATA_DIR = Path(__file__).resolve().parent
_TABLE_COUNTER = 0
DEFAULT_NO_FORMAT_COLS = {"Season", "PA", "BBE", "TBF", "IP"}
# Columns where higher values are worse (red=high, green=low) - inverted color scale
HIGHER_IS_WORSE_COLS = {
    "Hittable Pitch Take (%)",
    "Whiff vs. Secondaries (%)",
    "Whiff vs. 95+ (%)",
    "Ball (%)",
}


def ensure_streamlit() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return
    if get_script_run_ctx() is None:
        print("Run with: streamlit run damage_streamlit.py", file=sys.stderr)
        raise SystemExit(0)


ensure_streamlit()

st.set_page_config(page_title="Profiles", layout="wide")
st.markdown(
    """
    <style>
    .stDataFrame, .stDataFrame * {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def _load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


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
    # Prefer the most comprehensive file with newest data
    preferred_files = [
        DATA_DIR / "damage_pos_2015_2025.csv",
    ]
    for preferred in preferred_files:
        parquet_preferred = preferred.with_suffix(".parquet")
        if parquet_preferred.exists():
            return pd.read_parquet(parquet_preferred)
        if preferred.exists():
            return pd.read_csv(preferred)
    candidates = sorted(DATA_DIR.glob("damage_pos_*.parquet"))
    if candidates:
        return pd.read_parquet(candidates[-1])
    candidates = sorted(DATA_DIR.glob("damage_pos_*.csv"))
    if candidates:
        return pd.read_csv(candidates[-1])
    return pd.DataFrame()


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


def filter_by_values(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    if df.empty or "All" in values:
        return df
    return df[df[column].isin(values)]


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


def player_id_options(
    df: pd.DataFrame, id_col: str, name_col: str
) -> tuple[list, dict]:
    if df.empty or id_col not in df.columns:
        return ["All"], {}
    options_df = df[[id_col, name_col]].copy() if name_col in df.columns else df[[id_col]].copy()
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


def download_button(df: pd.DataFrame, label: str, key: str) -> None:
    if df.empty:
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


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _coerce_numeric_for_plot(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    plot_df = df.copy()
    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if plot_df.empty:
        return plot_df, numeric_cols

    min_required = max(3, int(len(plot_df) * 0.2))
    exclude_cols = {col.lower() for col in (exclude_cols or set())}
    object_cols = [
        col
        for col in plot_df.columns
        if col not in numeric_cols and pd.api.types.is_object_dtype(plot_df[col])
        and col.lower() not in exclude_cols
    ]
    for col in object_cols:
        cleaned = (
            plot_df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.notna().sum() >= min_required:
            plot_df[col] = coerced
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        for col in object_cols:
            if col in numeric_cols:
                continue
            cleaned = (
                plot_df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            coerced = pd.to_numeric(cleaned, errors="coerce")
            if coerced.notna().sum() >= 1:
                plot_df[col] = coerced
                numeric_cols.append(col)
                if len(numeric_cols) >= 2:
                    break

    return plot_df, numeric_cols


def _build_point_labels(
    df: pd.DataFrame,
    include_team: bool,
    label_cols: list[str] | None = None,
) -> pd.Series | None:
    if df.empty:
        return None
    player_col = _pick_first_col(
        df,
        [
            "Player",
            "player",
            "player_name",
            "Name",
            "Batter",
            "Pitcher",
            "Batter Name",
            "Pitcher Name",
        ],
    )
    team_col = None
    if include_team:
        team_col = _pick_first_col(
            df,
            [
                "Team",
                "team",
                "hitting_code",
                "pitching_code",
                "Team Code",
                "Team Abbrev",
            ],
        )
    pitch_col = _pick_first_col(
        df,
        [
            "Pitch Type",
            "pitch_type",
            "pitch_type_name",
            "PitchType",
            "TaggedPitchType",
            "pitch_tag",
        ],
    )
    split_col = _pick_first_col(
        df,
        [
            "split",
            "split_type",
            "Split",
            "Split Type",
        ],
    )
    if not any([player_col, team_col, pitch_col, split_col]):
        return None

    resolved_label_cols: list[str] = []
    if label_cols:
        lower_map = {col.lower(): col for col in df.columns}
        for name in label_cols:
            if name in df.columns:
                resolved_label_cols.append(name)
                continue
            key = name.lower()
            if key in lower_map:
                resolved_label_cols.append(lower_map[key])
    if include_team and team_col and team_col not in resolved_label_cols:
        resolved_label_cols.append(team_col)

    def build_label(row: pd.Series) -> str:
        if resolved_label_cols:
            parts: list[str] = []
            for col in resolved_label_cols:
                if not col:
                    continue
                value = row.get(col)
                if pd.isna(value):
                    continue
                value_str = str(value).strip()
                if value_str:
                    parts.append(value_str)
            return " | ".join(parts)

        if player_col and not include_team:
            parts: list[str] = []
            for col in [player_col, pitch_col, split_col]:
                if not col:
                    continue
                value = row.get(col)
                if pd.isna(value):
                    continue
                value_str = str(value).strip()
                if value_str:
                    parts.append(value_str)
            return " | ".join(parts)

        parts: list[str] = []
        for col in [player_col, team_col, pitch_col, split_col]:
            if not col:
                continue
            value = row.get(col)
            if pd.isna(value):
                continue
            value_str = str(value).strip()
            if value_str:
                parts.append(value_str)
        return " | ".join(parts)

    labels = df.apply(build_label, axis=1)
    if labels.str.strip().eq("").all():
        return None
    return labels


def _render_plot_controls(
    df: pd.DataFrame,
    table_key: str,
    include_team_label: bool,
    reverse_cols: set[str],
    label_cols: list[str] | None,
) -> None:
    exclude_cols = set(label_cols or [])
    plot_df, numeric_cols = _coerce_numeric_for_plot(df, exclude_cols=exclude_cols)
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns to plot.")
        return

    with st.expander("Create-a-Plot", expanded=False):
        col1, col2 = st.columns(2)
        x_col = col1.selectbox(
            "X column",
            options=numeric_cols,
            index=0,
            key=f"{table_key}_plot_x",
        )
        y_options = [col for col in numeric_cols if col != x_col]
        if not y_options:
            y_options = numeric_cols
        y_default = 1 if len(y_options) > 1 and y_options[0] == x_col else 0
        y_col = col2.selectbox(
            "Y column",
            options=y_options,
            index=y_default,
            key=f"{table_key}_plot_y",
        )
        col3, col4 = st.columns(2)
        size_options = ["(none)"] + numeric_cols
        size_col = col3.selectbox(
            "Size",
            options=size_options,
            index=0,
            key=f"{table_key}_plot_size",
        )
        color_options = ["(none)"] + numeric_cols
        color_col = col4.selectbox(
            "Color",
            options=color_options,
            index=0,
            key=f"{table_key}_plot_color",
        )
        max_points = st.number_input(
            "Max labeled points (sampled if exceeded)",
            min_value=100,
            max_value=20000,
            value=100,
            step=100,
            key=f"{table_key}_plot_max",
        )
        show_labels = st.checkbox(
            "Show point labels",
            value=True,
            key=f"{table_key}_plot_labels",
        )
        st.caption(
            "Large point counts will disable labels. Lower Max points to show labels."
        )

        plot_df = plot_df.copy()

        size_arg = None if size_col == "(none)" else size_col
        color_arg = None if color_col == "(none)" else color_col
        colorscale = None
        color_midpoint = None
        if color_arg is not None:
            colorscale = "RdYlGn_r" if color_arg in reverse_cols else "RdYlGn"
            color_midpoint = float(plot_df[color_arg].median())

        plot_df = plot_df.reset_index(drop=True)
        if x_col in plot_df.columns and y_col in plot_df.columns:
            plot_df = plot_df.copy()
            plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[x_col, y_col])
            if plot_df.empty:
                st.info("No rows available after numeric coercion of X/Y.")
                return
            q10_x = plot_df[x_col].quantile(0.1)
            q90_x = plot_df[x_col].quantile(0.9)
            q10_y = plot_df[y_col].quantile(0.1)
            q90_y = plot_df[y_col].quantile(0.9)
            extremes = (
                (plot_df[x_col] <= q10_x)
                | (plot_df[x_col] >= q90_x)
                | (plot_df[y_col] <= q10_y)
                | (plot_df[y_col] >= q90_y)
            )
            if show_labels:
                labels = _build_point_labels(
                    plot_df,
                    include_team=include_team_label,
                    label_cols=label_cols,
                )
                if labels is not None:
                    label_mask = extremes.copy()
                    if label_mask.sum() > max_points:
                        sampled_idx = (
                            plot_df[label_mask]
                            .sample(n=int(max_points), random_state=0)
                            .index
                        )
                        label_mask = plot_df.index.isin(sampled_idx)
                    plot_df = plot_df.copy()
                    plot_df["__label"] = labels.where(label_mask, "")

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            size=size_arg,
            color=color_arg,
            text="__label" if show_labels and "__label" in plot_df.columns else None,
            hover_name="__label" if "__label" in plot_df.columns else None,
            color_continuous_scale=colorscale,
            color_continuous_midpoint=color_midpoint,
            render_mode="svg",
        )
        if show_labels and "__label" in plot_df.columns:
            fig.update_traces(textposition="top center", mode="markers+text")
        else:
            fig.update_traces(mode="markers")
        fig.update_traces(marker=dict(size=7, opacity=0.7))
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_table(
    df: pd.DataFrame,
    reverse_cols: set[str] | None = None,
    no_format_cols: set[str] | None = None,
    group_cols: list[str] | None = None,
    stats_df: pd.DataFrame | None = None,
    show_controls: bool = True,
    include_team_label: bool = False,
    label_cols: list[str] | None = None,
) -> None:
    if df.empty:
        st.info("No data available yet.")
        return

    global _TABLE_COUNTER
    table_key = f"table_{_TABLE_COUNTER}"
    _TABLE_COUNTER += 1

    if show_controls:
        df = apply_column_filters(df, table_key)
        if df.empty:
            st.info("No data after filters.")
            return

    display_cols = [col for col in df.columns if not col.startswith("__")]
    df_display = df[display_cols].copy()

    _render_plot_controls(
        df_display,
        table_key,
        include_team_label,
        reverse_cols or set(),
        label_cols,
    )

    if show_controls:
        page_size_option = st.selectbox(
            "Rows per page",
            options=["All", 25, 50, 100, 200],
            index=2,
            key=f"{table_key}_page_size",
        )
        total_rows = len(df_display)
        if page_size_option == "All":
            page_size = total_rows
            page = 1
        else:
            page_size = int(page_size_option)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=int(total_pages),
                value=1,
                step=1,
                key=f"{table_key}_page",
            )

        start = (page - 1) * page_size
        end = start + page_size
        df_page_display = df_display.iloc[start:end].copy()
        df_page_full = df.iloc[start:end].copy()
    else:
        df_page_display = df_display.copy()
        df_page_full = df.copy()

    max_elements = pd.get_option("styler.render.max_elements")
    total_cells = df_page_display.shape[0] * df_page_display.shape[1]
    reverse_cols = reverse_cols or set()
    no_format_cols = no_format_cols or DEFAULT_NO_FORMAT_COLS
    numeric_cols = df_display.select_dtypes(include="number").columns
    float_cols = df.select_dtypes(include="floating").columns
    format_cols = [col for col in numeric_cols if col not in no_format_cols]

    if len(numeric_cols) > 0:
        df_page_display[numeric_cols] = df_page_display[numeric_cols].round(1)
    if len(float_cols) > 0:
        df_page_display[float_cols] = df_page_display[float_cols].round(1)

    if len(format_cols) > 0 and total_cells <= max_elements:
        stats_source = stats_df if stats_df is not None else df
        similarity_cols = [col for col in format_cols if col.startswith("Similarity")]
        stats_format_cols = [col for col in format_cols if col in stats_source.columns]
        if not stats_format_cols and not similarity_cols:
            st.dataframe(df_page_display, width="stretch", hide_index=True)
            return
        similarity_medians: dict[str, float] = {}
        for col in similarity_cols:
            if col in stats_source.columns:
                similarity_medians[col] = stats_source[col].median()
            else:
                similarity_medians[col] = df[col].median()
        group_cols = group_cols or []
        group_cols = [col for col in group_cols if col in stats_source.columns]
        if group_cols:
            if stats_format_cols:
                q10 = stats_source.groupby(group_cols)[stats_format_cols].quantile(0.05)
                q90 = stats_source.groupby(group_cols)[stats_format_cols].quantile(0.95)
                med = stats_source.groupby(group_cols)[stats_format_cols].median()
            else:
                q10 = q90 = med = None
        else:
            if stats_format_cols:
                q10 = stats_source[stats_format_cols].quantile(0.05)
                q90 = stats_source[stats_format_cols].quantile(0.95)
                med = stats_source[stats_format_cols].median()
            else:
                q10 = q90 = med = None
        cmap = colors.LinearSegmentedColormap.from_list(
            "rwgn", ["#c75c5c", "#f7f7f7", "#5cb85c"]
        )
        cmap_rev = colors.LinearSegmentedColormap.from_list(
            "gnrw", ["#5cb85c", "#f7f7f7", "#c75c5c"]
        )
        alpha = 0.9

        def style_row(row: pd.Series) -> list[str]:
            if group_cols:
                if q10 is None:
                    row_q10 = row_q90 = row_med = None
                else:
                    group_vals = df_page_full.loc[row.name, group_cols]
                    if isinstance(group_vals, pd.Series):
                        group_key = tuple(group_vals.values.tolist())
                    else:
                        group_key = group_vals
                    if group_key not in q10.index:
                        return [""] * len(row)
                    row_q10 = q10.loc[group_key]
                    row_q90 = q90.loc[group_key]
                    row_med = med.loc[group_key]
            else:
                row_q10 = q10
                row_q90 = q90
                row_med = med

            styles: list[str] = []
            for col in row.index:
                if col not in format_cols:
                    styles.append("")
                    continue
                if col in similarity_medians:
                    vmin = 0
                    vmax = 99
                    vcenter = similarity_medians[col]
                else:
                    if col not in stats_format_cols or row_q10 is None:
                        styles.append("")
                        continue
                    if row_q10 is None:
                        styles.append("")
                        continue
                    vmin = row_q10[col]
                    vmax = row_q90[col]
                    vcenter = row_med[col]
                if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
                    styles.append("")
                    continue
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                val = row[col]
                if pd.isna(val):
                    styles.append("")
                    continue
                val = float(np.clip(val, vmin, vmax))
                col_cmap = cmap_rev if col in reverse_cols else cmap
                rgb = colors.to_rgb(col_cmap(norm(val)))
                styles.append(
                    "background-color: "
                    f"rgba({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)},{alpha}); color: #000000"
                )
            return styles

        styler = df_page_display.style.apply(style_row, axis=1)
        if len(float_cols) > 0:
            format_map = {col: "{:.1f}" for col in float_cols}
            # Format integer-value columns without decimals
            int_keywords = ["Similarity", "Pitch Grade", "BB Spin", "Pctile", "#"]
            for col in df_page_display.columns:
                if any(kw in col for kw in int_keywords):
                    format_map[col] = "{:.0f}"
            styler = styler.format(format_map)
        st.dataframe(styler, width="stretch", hide_index=True)
        return
    if len(float_cols) > 0:
        # Identify columns that should display as integers
        int_keywords = ["Similarity", "Pitch Grade", "BB Spin", "Pctile", "#"]
        int_cols = [
            col
            for col in df_page_display.columns
            if any(kw in col for kw in int_keywords)
        ]
        other_float_cols = [col for col in float_cols if col not in int_cols]

        if other_float_cols:
            df_page_display[other_float_cols] = df_page_display[
                other_float_cols
            ].applymap(lambda x: f"{x:.1f}" if pd.notna(x) else x)
        for col in int_cols:
            if col in df_page_display.columns:
                df_page_display[col] = df_page_display[col].apply(
                    lambda x: f"{x:.0f}" if pd.notna(x) else x
                )
    st.dataframe(df_page_display, width="stretch", hide_index=True)


# Load datasets

damage_df = load_damage_df()
hitter_pct = load_csv("hitter_pctiles.csv")
pitcher_df = load_csv("pitcher_stuff_new.csv")
pitcher_pct = load_csv("pitcher_pctiles.csv")
hitting_avg = load_csv("new_hitting_lg_avg.csv")
pitching_avg = load_csv("new_lg_stuff.csv")
team_damage = load_csv("new_team_damage.csv")
team_stuff = load_csv("new_team_stuff.csv")
pitch_types = load_csv("new_pitch_types.csv")
pitch_types_pct = load_csv("pitch_types_pctiles.csv")
hitters_regressed = load_csv("hitters_regressed.csv")
pitchers_regressed = load_csv("pitchers_regressed.csv")
pitch_types_regressed = load_csv("pitch_types_regressed.csv")
hitter_splits_df = load_csv("hitter_splits.csv")
pitcher_splits_df = load_csv("pitcher_splits.csv")
pitch_type_splits_df = load_csv("pitch_types_splits.csv")
league_pitch_types = load_csv("league_pitch_types.csv")


# Normalize team column names: new CSVs use "team", old use "pitching_code"/"hitting_code"
def _normalize_team_col(df: pd.DataFrame, old_col: str) -> pd.DataFrame:
    """If 'team' column exists, rename it to old_col for backward compatibility."""
    if df.empty:
        return df
    if "team" in df.columns and old_col not in df.columns:
        return df.rename(columns={"team": old_col})
    return df


def _normalize_la_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename old FB_pct/GB_pct columns to new LA_gte_20/LA_lte_0 names."""
    if df.empty:
        return df
    rename_map = {}
    if "FB_pct" in df.columns and "LA_gte_20" not in df.columns:
        rename_map["FB_pct"] = "LA_gte_20"
    if "GB_pct" in df.columns and "LA_lte_0" not in df.columns:
        rename_map["GB_pct"] = "LA_lte_0"
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _normalize_split_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["split_type", "split"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _merge_regressed(
    base_df: pd.DataFrame, reg_df: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    if base_df.empty or reg_df.empty:
        return pd.DataFrame()
    reg_cols = [
        c
        for c in reg_df.columns
        if c.endswith("_reg") or c.endswith("_raw") or c.endswith("_n")
    ]
    keep_cols = list(dict.fromkeys(keys + reg_cols))
    reg_small = reg_df[keep_cols].drop_duplicates(subset=keys)
    return base_df.merge(reg_small, on=keys, how="left")


damage_df = _normalize_team_col(damage_df, "hitting_code")
damage_df = _normalize_la_cols(damage_df)
hitter_pct = _normalize_team_col(hitter_pct, "hitting_code")
hitter_pct = _normalize_la_cols(hitter_pct)
hitter_splits_df = _normalize_team_col(hitter_splits_df, "hitting_code")
hitter_splits_df = _normalize_la_cols(hitter_splits_df)
hitter_splits_df = _normalize_split_cols(hitter_splits_df)
pitcher_df = _normalize_team_col(pitcher_df, "pitching_code")
pitcher_df = _normalize_la_cols(pitcher_df)
pitcher_pct = _normalize_team_col(pitcher_pct, "pitching_code")
pitcher_splits_df = _normalize_team_col(pitcher_splits_df, "pitching_code")
pitcher_splits_df = _normalize_la_cols(pitcher_splits_df)
pitcher_splits_df = _normalize_split_cols(pitcher_splits_df)
pitch_types = _normalize_team_col(pitch_types, "pitching_code")
pitch_types_pct = _normalize_team_col(pitch_types_pct, "pitching_code")
pitch_type_splits_df = _normalize_team_col(pitch_type_splits_df, "pitching_code")
pitch_type_splits_df = _normalize_split_cols(pitch_type_splits_df)
league_pitch_types = _normalize_split_cols(league_pitch_types)
team_damage = _normalize_la_cols(team_damage)
team_stuff = _normalize_la_cols(team_stuff)

if (
    not pitch_types.empty
    and "pitch_group" not in pitch_types.columns
    and "pitch_tag" in pitch_types.columns
):
    pitch_types = pitch_types.assign(
        pitch_group=pitch_types["pitch_tag"].map(
            lambda tag: (
                "FA"
                if tag in {"FA", "HC", "SI"}
                else (
                    "BR"
                    if tag in {"SL", "SW", "CU"}
                    else "OFF" if tag in {"CH", "FS"} else "OTHER"
                )
            )
        )
    )

if (
    not pitch_type_splits_df.empty
    and "pitch_group" not in pitch_type_splits_df.columns
    and "pitch_tag" in pitch_type_splits_df.columns
):
    pitch_type_splits_df = pitch_type_splits_df.assign(
        pitch_group=pitch_type_splits_df["pitch_tag"].map(
            lambda tag: (
                "FA"
                if tag in {"FA", "HC", "SI"}
                else (
                    "BR"
                    if tag in {"SL", "SW", "CU"}
                    else "OFF" if tag in {"CH", "FS"} else "OTHER"
                )
            )
        )
    )

hitters_reg_df = _merge_regressed(
    damage_df,
    hitters_regressed,
    ["batter_mlbid", "hitter_name", "season", "level_id"],
)
pitchers_reg_df = _merge_regressed(
    pitcher_df,
    pitchers_regressed,
    ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand"],
)
pitch_types_reg_df = _merge_regressed(
    pitch_types,
    pitch_types_regressed,
    ["pitcher_mlbid", "name", "pitcher_hand", "season", "level_id", "pitch_tag"],
)


# =============================================================================
# PAGE FUNCTIONS
# =============================================================================


def home_page():
    """Welcome/Home page"""
    st.title("Profiles")

    st.markdown(
        """
Welcome! Here you will find metrics I (https://twitter.com/NotTheBobbyOrr) have developed for analyzing hitters & pitchers at a player and team level.
I make frequent use of these statistics in my work at BaseballProspectus dot com (https://www.baseballprospectus.com/author/ringtheodubel/) and for my own fantasy strategy.
"""
    )

    st.markdown("---")
    st.subheader("Welcome to Premium Features")
    st.markdown(
        """
Navigate via the sidebar to explore different analytics tools. There are glossaries containing explanations for each statistic.
"""
    )
    st.markdown(
        """
Feedback: If you have any suggestions or just want to say hi, shoot me a DM on Twitter or send me an email at orrrobf @ gmail dot com.
"""
    )
    st.write(f"Last Update: {pd.Timestamp.today().date()}")


# =============================================================================
# HITTERS PAGES
# =============================================================================


def hitter_individual_stats():
    """Hitters - Individual Stats page"""
    st.title("Individual Hitter Stats")

    if damage_df.empty:
        st.info("Missing damage_pos_2015_2025.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="hitter_stats_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(damage_df),
                default=(
                    [season_options(damage_df)[1]]
                    if len(season_options(damage_df)) > 1
                    else ["All"]
                ),
                key="hitter_stats_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
                key="hitter_stats_min_value",
            )
            value_type = st.selectbox(
                "Filter By", ["PA", "BBE"], index=1, key="hitter_stats_value_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(damage_df, "hitting_code"),
                index=0,
                key="hitter_stats_team",
            )
            player_options, player_name_map = player_id_options(
                damage_df, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="hitter_stats_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = damage_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = damage_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_values(df, "batter_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name",
                "hitting_code",
                "season",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "max_EV",
                "pull_FB_pct",
                "LA_gte_20",
                "LA_lte_0",
                "SEAGER",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "whiffs_vs_95",
                "contact_vs_avg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name",
                "hitting_code": "Team",
                "season": "Season",
                "bbe": "BBE",
                "damage_rate": "Damage/BBE (%)",
                "EV90th": "90th Pctile EV",
                "max_EV": "Max EV",
                "pull_FB_pct": "Pulled FB (%)",
                "LA_gte_20": "LA>=20%",
                "LA_lte_0": "LA<=0%",
                "selection_skill": "Selectivity (%)",
                "hittable_pitches_taken": "Hittable Pitch Take (%)",
                "chase": "Chase (%)",
                "z_con": "Z-Contact (%)",
                "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                "whiffs_vs_95": "Whiff vs. 95+ (%)",
                "contact_vs_avg": "Contact Over Expected (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols=HIGHER_IS_WORSE_COLS | {"Chase (%)", "LA<=0%"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                include_team_label=False,
            )
            download_button(df, "hitters", "hitters_download")


def hitter_percentiles():
    """Hitters - Percentiles page"""
    st.title("Hitter Percentiles")

    if hitter_pct.empty:
        st.info("Missing hitter_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="hitter_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(hitter_pct),
                default=(
                    [season_options(hitter_pct)[1]]
                    if len(season_options(hitter_pct)) > 1
                    else ["All"]
                ),
                key="hitter_pct_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
                key="hitter_pct_min_value",
            )
            value_type = st.selectbox(
                "Filter By", ["PA", "BBE"], index=1, key="hitter_pct_value_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(hitter_pct, "hitting_code"),
                index=0,
                key="hitter_pct_team",
            )
            player_options, player_name_map = player_id_options(
                hitter_pct, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="hitter_pct_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = hitter_pct.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_values(df, "batter_mlbid", player)

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name",
                "season",
                "hitting_code",
                "SEAGER_pctile",
                "selection_skill_pctile",
                "hittable_pitches_taken_pctile",
                "damage_rate_pctile",
                "EV90th_pctile",
                "max_EV_pctile",
                "pull_FB_pct_pctile",
                "chase_pctile",
                "z_con_pctile",
                "secondary_whiff_pct_pctile",
                "whiffs_vs_95_pctile",
                "contact_vs_avg_pctile",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name",
                "hitting_code": "Team",
                "season": "Season",
                "SEAGER_pctile": "SEAGER",
                "selection_skill_pctile": "Selection Skill",
                "hittable_pitches_taken_pctile": "Hittable Pitch Take",
                "damage_rate_pctile": "Damage Rate",
                "EV90th_pctile": "90th Pctile EV",
                "max_EV_pctile": "Max EV",
                "pull_FB_pct_pctile": "Pulled FB",
                "chase_pctile": "Chase",
                "z_con_pctile": "Z-Contact",
                "secondary_whiff_pct_pctile": "Whiff vs Secondaries",
                "whiffs_vs_95_pctile": "Whiff vs 95+",
                "contact_vs_avg_pctile": "Contact Over Expected",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage Rate", ascending=False)
            # For percentiles, reverse color on bad stats (higher pctile in bad stat = worse)
            render_table(
                df,
                reverse_cols={
                    "Hittable Pitch Take",
                    "Chase",
                    "Whiff vs Secondaries",
                    "Whiff vs 95+",
                },
            )
            download_button(df, "hitter_percentiles", "hitter_pct_download")


def hitter_comps():
    """Hitters - Comparisons page"""
    st.title("Hitter Comparisons (Auto-Regressed)")

    if hitters_reg_df.empty:
        st.info("Missing hitters_regressed.csv")
    else:
        player_pool = hitters_reg_df.copy()
        player_pool = player_pool[
            (player_pool["level_id"] == 1) & (player_pool["PA"] >= 20)
        ]
        eligible_all = hitters_reg_df.copy()
        eligible_all = eligible_all[
            (eligible_all["level_id"] == 1) & (eligible_all["PA"] >= 200)
        ]
        if player_pool.empty:
            st.info("No eligible MLB hitter seasons (min 20 PA).")
        else:
            seasons = season_options(player_pool, "season")[1:]
            season_choice = st.selectbox(
                "Season", seasons, index=0, key="hitter_comps_season"
            )
            season_df = player_pool[player_pool["season"] == season_choice]
            player_options, player_name_map = player_id_options(
                season_df, "batter_mlbid", "hitter_name"
            )
            player_choice = st.selectbox(
                "Player",
                [opt for opt in player_options if opt != "All"],
                index=0,
                format_func=lambda v: f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="hitter_comps_player",
            )
            player_df = season_df[season_df["batter_mlbid"] == player_choice]
            player_all = hitters_reg_df[hitters_reg_df["batter_mlbid"] == player_choice]
            teams = team_options(player_all, "hitting_code")[1:]
            team_choice = (
                st.selectbox("Team", teams, index=0, key="hitter_comps_team")
                if len(teams) > 1
                else (teams[0] if teams else None)
            )
            if team_choice:
                player_df = filter_by_team_token(player_df, "hitting_code", team_choice)

            feature_cols = [
                "damage_rate_reg",
                "EV90th_reg",
                "pull_FB_pct_reg",
                "LA_gte_20_reg",
                "LA_lte_0_reg",
                "SEAGER_reg",
                "selection_skill_reg",
                "hittable_pitches_taken_reg",
                "chase_reg",
                "z_con_reg",
                "secondary_whiff_pct_reg",
                "whiffs_vs_95_reg",
                "contact_vs_avg_reg",
            ]
            feature_cols = [c for c in feature_cols if c in eligible_all.columns]
            eligible_comp = eligible_all.dropna(subset=feature_cols)
            if player_df.empty:
                st.info("No season row found for that selection.")
            else:
                eligible_comp = eligible_comp[
                    ~(eligible_comp["batter_mlbid"] == player_choice)
                ]
                stats = eligible_comp[feature_cols]
                means = stats.mean()
                stds = stats.std(ddof=0).replace(0, np.nan)
                zscores = (stats - means) / stds
                zscores = zscores.fillna(0)
                target_vec = (
                    ((player_df[feature_cols] - means) / stds)
                    .fillna(0)
                    .iloc[0]
                    .to_numpy()
                )
                distances = np.linalg.norm(zscores.to_numpy() - target_vec, axis=1)
                max_dist = distances.max() if len(distances) else 0.0
                if max_dist == 0:
                    similarity = np.full_like(distances, 100.0, dtype=float)
                else:
                    similarity = 100 * (1 - (distances / max_dist))
                eligible_comp = eligible_comp.copy()
                eligible_comp["similarity_score"] = similarity.round(0)
                eligible_comp = eligible_comp.sort_values(
                    "similarity_score", ascending=False
                )

                display_cols = [
                    "hitter_name",
                    "hitting_code",
                    "season",
                    "PA",
                    "bbe",
                    "similarity_score",
                    *feature_cols,
                ]
                eligible_comp = eligible_comp.assign(
                    __season=eligible_comp["season"], __level=eligible_comp["level_id"]
                )
                display_cols += ["__season", "__level"]
                df = eligible_comp[display_cols].copy()
                df = df.rename(
                    columns={
                        "hitter_name": "Name",
                        "hitting_code": "Team",
                        "season": "Season",
                        "bbe": "BBE",
                        "damage_rate_reg": "Damage/BBE (%)",
                        "EV90th_reg": "90th Pctile EV",
                        "pull_FB_pct_reg": "Pulled FB (%)",
                        "selection_skill_reg": "Selectivity (%)",
                        "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
                        "chase_reg": "Chase (%)",
                        "z_con_reg": "Z-Contact (%)",
                        "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
                        "similarity_score": "Similarity (0-100)",
                        "LA_gte_20_reg": "LA>=20%",
                        "LA_lte_0_reg": "LA<=0%",
                        "SEAGER_reg": "SEAGER",
                        "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
                        "contact_vs_avg_reg": "Contact Over Expected (%)",
                    }
                )
                stats_df = hitters_reg_df.copy()
                stats_df = stats_df.assign(
                    __season=stats_df["season"], __level=stats_df["level_id"]
                )
                stats_columns = [
                    "hitter_name",
                    "hitting_code",
                    "season",
                    "PA",
                    "bbe",
                    "damage_rate_reg",
                    "EV90th_reg",
                    "pull_FB_pct_reg",
                    "LA_gte_20_reg",
                    "LA_lte_0_reg",
                    "SEAGER_reg",
                    "selection_skill_reg",
                    "hittable_pitches_taken_reg",
                    "chase_reg",
                    "z_con_reg",
                    "secondary_whiff_pct_reg",
                    "whiffs_vs_95_reg",
                    "contact_vs_avg_reg",
                    "__season",
                    "__level",
                ]
                stats_df = stats_df[
                    [col for col in stats_columns if col in stats_df.columns]
                ].rename(
                    columns={
                        "hitter_name": "Name",
                        "hitting_code": "Team",
                        "season": "Season",
                        "bbe": "BBE",
                        "damage_rate_reg": "Damage/BBE (%)",
                        "EV90th_reg": "90th Pctile EV",
                        "pull_FB_pct_reg": "Pulled FB (%)",
                        "LA_gte_20_reg": "LA>=20%",
                        "LA_lte_0_reg": "LA<=0%",
                        "SEAGER_reg": "SEAGER",
                        "selection_skill_reg": "Selectivity (%)",
                        "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
                        "chase_reg": "Chase (%)",
                        "z_con_reg": "Z-Contact (%)",
                        "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
                        "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
                        "contact_vs_avg_reg": "Contact Over Expected (%)",
                    }
                )
                target_display_cols = [
                    "hitter_name",
                    "hitting_code",
                    "season",
                    "PA",
                    "bbe",
                    "damage_rate_reg",
                    "EV90th_reg",
                    "pull_FB_pct_reg",
                    "LA_gte_20_reg",
                    "LA_lte_0_reg",
                    "SEAGER_reg",
                    "selection_skill_reg",
                    "hittable_pitches_taken_reg",
                    "chase_reg",
                    "z_con_reg",
                    "secondary_whiff_pct_reg",
                    "whiffs_vs_95_reg",
                    "contact_vs_avg_reg",
                    "__season",
                    "__level",
                ]
                target_df = player_df.assign(
                    __season=player_df["season"], __level=player_df["level_id"]
                )
                target_df = target_df[
                    [col for col in target_display_cols if col in target_df.columns]
                ].copy()
                target_df = target_df.rename(
                    columns={
                        "hitter_name": "Name",
                        "hitting_code": "Team",
                        "season": "Season",
                        "bbe": "BBE",
                        "damage_rate_reg": "Damage/BBE (%)",
                        "EV90th_reg": "90th Pctile EV",
                        "pull_FB_pct_reg": "Pulled FB (%)",
                        "selection_skill_reg": "Selectivity (%)",
                        "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
                        "chase_reg": "Chase (%)",
                        "z_con_reg": "Z-Contact (%)",
                        "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
                        "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
                        "LA_gte_20_reg": "LA>=20%",
                        "LA_lte_0_reg": "LA<=0%",
                        "SEAGER_reg": "SEAGER",
                        "contact_vs_avg_reg": "Contact Over Expected (%)",
                    }
                )
                st.caption("Selected season")
                render_table(
                    target_df,
                    reverse_cols=HIGHER_IS_WORSE_COLS | {"LA<=0%", "Chase (%)"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                    show_controls=False,
                )
                st.caption("Most similar MLB seasons (PA >= 200)")
                render_table(
                    df,
                    reverse_cols=HIGHER_IS_WORSE_COLS | {"LA<=0%", "Chase (%)"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                )


def hitter_ar():
    """Hitters - Auto Regressed page"""
    st.title("Hitters - Auto Regressed")

    if hitters_reg_df.empty:
        st.info("Missing hitters_regressed.csv or damage_pos_2015_2025.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="hitter_ar_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(hitters_reg_df),
                default=(
                    [season_options(hitters_reg_df)[1]]
                    if len(season_options(hitters_reg_df)) > 1
                    else ["All"]
                ),
                key="hitter_ar_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
                key="hitter_ar_min_value",
            )
            value_type = st.selectbox(
                "Filter By", ["PA", "BBE"], index=1, key="hitter_ar_value_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(hitters_reg_df, "hitting_code"),
                index=0,
                key="hitter_ar_team",
            )
            player_options, player_name_map = player_id_options(
                hitters_reg_df, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="hitter_ar_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = hitters_reg_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = hitters_reg_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_values(df, "batter_mlbid", player)

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name",
                "hitting_code",
                "season",
                "PA",
                "bbe",
                "damage_rate_reg",
                "EV90th_reg",
                "max_EV_reg",
                "pull_FB_pct_reg",
                "LA_gte_20_reg",
                "LA_lte_0_reg",
                "SEAGER_reg",
                "selection_skill_reg",
                "hittable_pitches_taken_reg",
                "chase_reg",
                "z_con_reg",
                "secondary_whiff_pct_reg",
                "whiffs_vs_95_reg",
                "contact_vs_avg_reg",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name",
                "hitting_code": "Team",
                "season": "Season",
                "bbe": "BBE",
                "damage_rate_reg": "Damage/BBE (%)",
                "EV90th_reg": "90th Pctile EV",
                "max_EV_reg": "Max EV",
                "pull_FB_pct_reg": "Pulled FB (%)",
                "LA_gte_20_reg": "LA>=20%",
                "LA_lte_0_reg": "LA<=0%",
                "SEAGER_reg": "SEAGER",
                "selection_skill_reg": "Selectivity (%)",
                "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
                "chase_reg": "Chase (%)",
                "z_con_reg": "Z-Contact (%)",
                "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
                "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
                "contact_vs_avg_reg": "Contact Over Expected (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols=HIGHER_IS_WORSE_COLS | {"LA<=0%", "Chase (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "hitters_ar", "hitters_ar_download")


def hitter_splits():
    """Hitters - Splits page (placeholder)"""
    st.title("Hitter Splits")

    if hitter_splits_df.empty:
        st.info("Missing hitter_splits.csv")
        return

    tabs = st.tabs(["vL / vR", "Home / Away", "1H / 2H", "Monthly"])
    split_map = {
        "vL / vR": "vs L/R",
        "Home / Away": "Home/Away",
        "1H / 2H": "1st Half/2nd Half",
        "Monthly": "Monthly",
    }

    for idx, tab_name in enumerate(split_map.keys()):
        with tabs[idx]:
            split_type = split_map[tab_name]
            split_df = hitter_splits_df[
                hitter_splits_df["split_type"] == split_type
            ].copy()
            if split_df.empty:
                available = sorted(
                    hitter_splits_df["split_type"].dropna().unique().tolist()
                )
                st.info(f"No data for {tab_name}. Available split types: {available}")
                continue

            left, right = st.columns([1, 3])
            with left:
                level = st.selectbox(
                    "Select Level",
                    ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                    index=1,
                    key=f"hitter_splits_level_{idx}",
                )
                season = st.multiselect(
                    "Select Season",
                    season_options(split_df),
                    default=(
                        [season_options(split_df)[1]]
                        if len(season_options(split_df)) > 1
                        else ["All"]
                    ),
                    key=f"hitter_splits_season_{idx}",
                )
                min_value = st.number_input(
                    "Minimum Value",
                    min_value=0,
                    max_value=500,
                    value=100,
                    step=1,
                    key=f"hitter_splits_min_value_{idx}",
                )
                value_type = st.selectbox(
                    "Filter By",
                    ["PA", "BBE"],
                    index=1,
                    key=f"hitter_splits_value_type_{idx}",
                )
                split_choice = st.multiselect(
                    "Select Split",
                    ["All"] + sorted(split_df["split"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"hitter_splits_split_{idx}",
                )
                team = st.selectbox(
                    "Select Team",
                    team_options(split_df, "hitting_code"),
                    index=0,
                    key=f"hitter_splits_team_{idx}",
                )
                player_options, player_name_map = player_id_options(
                    split_df, "batter_mlbid", "hitter_name"
                )
                player = st.multiselect(
                    "Select Player",
                    player_options,
                    default=["All"],
                    format_func=lambda v: "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                    key=f"hitter_splits_player_{idx}",
                )
            with right:
                level_map = {
                    "All": [1, 11, 14, 16],
                    "MLB": [1],
                    "Triple-A": [11],
                    "Low-A": [14],
                    "Low Minors": [16],
                }
                base_stats = split_df.copy()
                base_stats = base_stats.assign(
                    __season=base_stats["season"],
                    __level=base_stats["level_id"],
                )
                df = split_df.copy()
                df = df[df["level_id"].isin(level_map[level])]
                df = filter_by_values(df, "season", season)
                df = filter_by_values(df, "split", split_choice)
                df = filter_by_team_token(df, "hitting_code", team)
                df = filter_by_values(df, "batter_mlbid", player)
                df = df.assign(__season=df["season"], __level=df["level_id"])

                if value_type == "PA":
                    df = numeric_filter(df, "PA", min_value)
                else:
                    df = numeric_filter(df, "bbe", min_value)

                columns = [
                    "hitter_name",
                    "hitting_code",
                    "season",
                    "split",
                    "PA",
                    "bbe",
                    "damage_rate",
                    "EV90th",
                    "max_EV",
                    "pull_FB_pct",
                    "LA_gte_20",
                    "LA_lte_0",
                    "SEAGER",
                    "selection_skill",
                    "hittable_pitches_taken",
                    "chase",
                    "z_con",
                    "secondary_whiff_pct",
                    "whiffs_vs_95",
                    "contact_vs_avg",
                    "__season",
                    "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                rename_map = {
                    "hitter_name": "Name",
                    "hitting_code": "Team",
                    "season": "Season",
                    "split": "Split",
                    "bbe": "BBE",
                    "damage_rate": "Damage/BBE (%)",
                    "EV90th": "90th Pctile EV",
                    "max_EV": "Max EV",
                    "pull_FB_pct": "Pulled FB (%)",
                    "LA_gte_20": "LA>=20%",
                    "LA_lte_0": "LA<=0%",
                    "selection_skill": "Selectivity (%)",
                    "hittable_pitches_taken": "Hittable Pitch Take (%)",
                    "chase": "Chase (%)",
                    "z_con": "Z-Contact (%)",
                    "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                    "whiffs_vs_95": "Whiff vs. 95+ (%)",
                    "contact_vs_avg": "Contact Over Expected (%)",
                }
                df = df.rename(columns=rename_map)
                df = df.sort_values(by="Damage/BBE (%)", ascending=False)
                stats_df = base_stats[
                    [col for col in columns if col in base_stats.columns]
                ].rename(columns=rename_map)
                render_table(
                    df,
                    reverse_cols=HIGHER_IS_WORSE_COLS | {"Chase (%)", "LA<=0%"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                )
                download_button(
                    df,
                    f"hitter_splits_{idx}",
                    f"hitter_splits_download_{idx}",
                )


# =============================================================================
# PITCHERS PAGES
# =============================================================================


def pitcher_individual_stats():
    """Pitchers - Individual Stats page"""
    st.title("Individual Pitcher Stats")

    if pitcher_df.empty:
        st.info("Missing pitcher_stuff_new.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitcher_stats_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitcher_df),
                default=(
                    [season_options(pitcher_df)[1]]
                    if len(season_options(pitcher_df)) > 1
                    else ["All"]
                ),
                key="pitcher_stats_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=1000,
                value=100,
                step=1,
                key="pitcher_stats_min_value",
            )
            filter_type = st.selectbox(
                "Filter By", ["IP", "TBF"], index=1, key="pitcher_stats_filter_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitcher_df, "pitching_code"),
                index=0,
                key="pitcher_stats_team",
            )
            player_options, player_name_map = player_id_options(
                pitcher_df, "pitcher_mlbid", "name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="pitcher_stats_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitcher_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitcher_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            if filter_type == "IP":
                df = numeric_filter(df, "IP", min_value)
            else:
                df = numeric_filter(df, "TBF", min_value)

            columns = [
                "name",
                "season",
                "pitching_code",
                "TBF",
                "IP",
                "stuff",
                "fastball_velo",
                "max_velo",
                "fastball_vaa",
                "FA_pct",
                "BB_rpm",
                "SwStr",
                "Zone",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "CSW",
                "LA_lte_0",
                "rel_z",
                "rel_x",
                "ext",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round BB_rpm and stuff to integers
            if "BB_rpm" in df.columns:
                df["BB_rpm"] = df["BB_rpm"].round(0)
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            rename_map = {
                "name": "Name",
                "pitching_code": "Team",
                "season": "Season",
                "stuff": "Pitch Grade",
                "fastball_velo": "FA mph",
                "max_velo": "Max FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "BB_rpm": "BB Spin",
                "SwStr": "SwStr (%)",
                "Zone": "Zone (%)",
                "Ball_pct": "Ball (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "CSW": "CSW (%)",
                "LA_lte_0": "LA<=0%",
                "rel_z": "Vertical Release (ft.)",
                "rel_x": "Horizontal Release (ft.)",
                "ext": "Extension (ft.)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "pitchers", "pitchers_download")


def pitcher_percentiles():
    """Pitchers - Percentiles page"""
    st.title("Pitcher Percentiles")

    if pitcher_pct.empty:
        st.info("Missing pitcher_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitcher_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitcher_pct),
                default=(
                    [season_options(pitcher_pct)[1]]
                    if len(season_options(pitcher_pct)) > 1
                    else ["All"]
                ),
                key="pitcher_pct_season",
            )
            min_value = st.number_input(
                "Minimum TBF",
                min_value=0,
                max_value=1000,
                value=100,
                step=1,
                key="pitcher_pct_min_value",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitcher_pct, "pitching_code"),
                index=0,
                key="pitcher_pct_team",
            )
            player_options, player_name_map = player_id_options(
                pitcher_pct, "pitcher_mlbid", "name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="pitcher_pct_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = pitcher_pct.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = numeric_filter(df, "TBF", min_value)

            columns = [
                "name",
                "season",
                "pitching_code",
                "stuff_pctile",
                "fastball_velo_pctile",
                "max_velo_pctile",
                "fastball_vaa_pctile",
                "SwStr_pctile",
                "Ball_pct_pctile",
                "Z_Contact_pctile",
                "Chase_pctile",
                "CSW_pctile",
                "rel_z_pctile",
                "rel_x_pctile",
                "ext_pctile",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitching_code": "Team",
                "season": "Season",
                "stuff_pctile": "Pitch Grade Pctile",
                "fastball_velo_pctile": "Avg FA mph",
                "max_velo_pctile": "Max FA mph",
                "fastball_vaa_pctile": "FA VAA",
                "SwStr_pctile": "SwStr (%)",
                "Ball_pct_pctile": "Ball (%)",
                "Z_Contact_pctile": "Z-Contact (%)",
                "Chase_pctile": "Chase (%)",
                "CSW_pctile": "CSW (%)",
                "rel_z_pctile": "Vertical Release (ft.)",
                "rel_x_pctile": "Horizontal Release (ft.)",
                "ext_pctile": "Extension (ft.)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            render_table(df, reverse_cols={"FA VAA", "Ball (%)", "Z-Contact (%)"})
            download_button(df, "pitcher_percentiles", "pitcher_pct_download")


def pitcher_comps():
    """Pitchers - Comparisons page"""
    st.title("Pitcher Comparisons (Auto-Regressed)")

    if pitchers_reg_df.empty:
        st.info("Missing pitchers_regressed.csv or pitcher_stuff_new.csv")
    else:
        player_pool = pitchers_reg_df.copy()
        player_pool = player_pool[
            (player_pool["level_id"] == 1) & (player_pool["IP"] >= 5)
        ]
        eligible_all = pitchers_reg_df.copy()
        eligible_all = eligible_all[
            (eligible_all["level_id"] == 1) & (eligible_all["IP"] >= 50)
        ]
        if player_pool.empty:
            st.info("No eligible MLB pitcher seasons (min 5 IP).")
        else:
            seasons = season_options(player_pool, "season")[1:]
            season_choice = st.selectbox(
                "Season", seasons, index=0, key="pitcher_comps_season"
            )
            season_df = player_pool[player_pool["season"] == season_choice]
            player_options, player_name_map = player_id_options(
                season_df, "pitcher_mlbid", "name"
            )
            player_choice = st.selectbox(
                "Player",
                [opt for opt in player_options if opt != "All"],
                index=0,
                format_func=lambda v: f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="pitcher_comps_player",
            )
            player_df = season_df[season_df["pitcher_mlbid"] == player_choice]
            player_all = pitchers_reg_df[
                pitchers_reg_df["pitcher_mlbid"] == player_choice
            ]
            teams = team_options(player_all, "pitching_code")[1:]
            team_choice = (
                st.selectbox("Team", teams, index=0, key="pitcher_comps_team")
                if len(teams) > 1
                else (teams[0] if teams else None)
            )
            if team_choice:
                player_df = filter_by_team_token(player_df, "pitching_code", team_choice)

            feature_cols = [
                "stuff",
                "fastball_velo_reg",
                "fastball_vaa_reg",
                "FA_pct_reg",
                "BB_rpm_reg",
                "SwStr_reg",
                "Ball_pct_reg",
                "Z_Contact_reg",
                "Chase_reg",
                "LA_lte_0_reg",
                "rel_z_reg",
                "rel_x_reg",
                "ext_reg",
            ]
            feature_cols = [c for c in feature_cols if c in eligible_all.columns]
            eligible_comp = eligible_all.dropna(subset=feature_cols)
            if player_df.empty:
                st.info("No season row found for that selection.")
            else:
                eligible_comp = eligible_comp[
                    ~(eligible_comp["pitcher_mlbid"] == player_choice)
                ]
                stats = eligible_comp[feature_cols]
                means = stats.mean()
                stds = stats.std(ddof=0).replace(0, np.nan)
                zscores = (stats - means) / stds
                zscores = zscores.fillna(0)
                target_vec = (
                    ((player_df[feature_cols] - means) / stds)
                    .fillna(0)
                    .iloc[0]
                    .to_numpy()
                )
                distances = np.linalg.norm(zscores.to_numpy() - target_vec, axis=1)
                max_dist = distances.max() if len(distances) else 0.0
                if max_dist == 0:
                    similarity = np.full_like(distances, 100.0, dtype=float)
                else:
                    similarity = 100 * (1 - (distances / max_dist))
                eligible_comp = eligible_comp.copy()
                eligible_comp["similarity_score"] = similarity.round(0)
                eligible_comp = eligible_comp.sort_values(
                    "similarity_score", ascending=False
                )

                display_cols = [
                    "name",
                    "pitching_code",
                    "season",
                    "TBF",
                    "IP",
                    "similarity_score",
                    *feature_cols,
                ]
                if "stuff_z" in eligible_comp.columns:
                    display_cols.insert(6, "stuff_z")

                eligible_comp = eligible_comp.assign(
                    __season=eligible_comp["season"], __level=eligible_comp["level_id"]
                )
                display_cols += ["__season", "__level"]
                df = eligible_comp[
                    [col for col in display_cols if col in eligible_comp.columns]
                ].copy()
                df = df.rename(
                    columns={
                        "name": "Name",
                        "pitching_code": "Team",
                        "season": "Season",
                        "fastball_velo_reg": "FA mph",
                        "fastball_vaa_reg": "FA VAA",
                        "SwStr_reg": "SwStr (%)",
                        "Ball_pct_reg": "Ball (%)",
                        "Chase_reg": "Chase (%)",
                        "Z_Contact_reg": "Z-Contact (%)",
                        "LA_lte_0_reg": "LA<=0%",
                        "rel_z_reg": "Vertical Release (ft.)",
                        "rel_x_reg": "Horizontal Release (ft.)",
                        "ext_reg": "Extension (ft.)",
                        "similarity_score": "Similarity (0-100)",
                        "stuff": "Pitch Grade",
                        "stuff_z": "Pitch Grade Z",
                        "FA_pct_reg": "FA Usage (%)",
                        "BB_rpm_reg": "BB Spin",
                    }
                )
                stats_df = pitchers_reg_df.copy()
                stats_df = stats_df.assign(
                    __season=stats_df["season"], __level=stats_df["level_id"]
                )
                stats_columns = [
                    "name",
                    "pitching_code",
                    "season",
                    "TBF",
                    "IP",
                    "stuff",
                    "fastball_velo_reg",
                    "fastball_vaa_reg",
                    "FA_pct_reg",
                    "BB_rpm_reg",
                    "SwStr_reg",
                    "Ball_pct_reg",
                    "Z_Contact_reg",
                    "Chase_reg",
                    "LA_lte_0_reg",
                    "rel_z_reg",
                    "rel_x_reg",
                    "ext_reg",
                    "__season",
                    "__level",
                ]
                stats_df = stats_df[
                    [col for col in stats_columns if col in stats_df.columns]
                ].rename(
                    columns={
                        "name": "Name",
                        "pitching_code": "Team",
                        "season": "Season",
                        "fastball_velo_reg": "FA mph",
                        "fastball_vaa_reg": "FA VAA",
                        "SwStr_reg": "SwStr (%)",
                        "Ball_pct_reg": "Ball (%)",
                        "Chase_reg": "Chase (%)",
                        "Z_Contact_reg": "Z-Contact (%)",
                        "LA_lte_0_reg": "LA<=0%",
                        "rel_z_reg": "Vertical Release (ft.)",
                        "rel_x_reg": "Horizontal Release (ft.)",
                        "ext_reg": "Extension (ft.)",
                        "stuff": "Pitch Grade",
                        "FA_pct_reg": "FA Usage (%)",
                        "BB_rpm_reg": "BB Spin",
                    }
                )
                target_display_cols = [
                    "name",
                    "pitching_code",
                    "season",
                    "TBF",
                    "IP",
                    "stuff",
                    "fastball_velo_reg",
                    "fastball_vaa_reg",
                    "FA_pct_reg",
                    "BB_rpm_reg",
                    "SwStr_reg",
                    "Ball_pct_reg",
                    "Z_Contact_reg",
                    "Chase_reg",
                    "LA_lte_0_reg",
                    "rel_z_reg",
                    "rel_x_reg",
                    "ext_reg",
                    "__season",
                    "__level",
                ]
                target_df = player_df.assign(
                    __season=player_df["season"], __level=player_df["level_id"]
                )
                target_df = target_df[
                    [col for col in target_display_cols if col in target_df.columns]
                ].copy()
                target_df = target_df.rename(
                    columns={
                        "name": "Name",
                        "pitching_code": "Team",
                        "season": "Season",
                        "fastball_velo_reg": "FA mph",
                        "fastball_vaa_reg": "FA VAA",
                        "SwStr_reg": "SwStr (%)",
                        "Ball_pct_reg": "Ball (%)",
                        "Chase_reg": "Chase (%)",
                        "Z_Contact_reg": "Z-Contact (%)",
                        "LA_lte_0_reg": "LA<=0%",
                        "rel_z_reg": "Vertical Release (ft.)",
                        "rel_x_reg": "Horizontal Release (ft.)",
                        "ext_reg": "Extension (ft.)",
                        "stuff": "Pitch Grade",
                        "FA_pct_reg": "FA Usage (%)",
                        "BB_rpm_reg": "BB Spin",
                    }
                )
                st.caption("Selected season")
                render_table(
                    target_df,
                    reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                    show_controls=False,
                )
                st.caption("Most similar MLB seasons (IP >= 50)")
                render_table(
                    df,
                    reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                )


def pitcher_ar():
    """Pitchers - Auto Regressed page"""
    st.title("Pitchers - Auto Regressed")

    if pitchers_reg_df.empty:
        st.info("Missing pitchers_regressed.csv or pitcher_stuff_new.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitcher_ar_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitchers_reg_df),
                default=(
                    [season_options(pitchers_reg_df)[1]]
                    if len(season_options(pitchers_reg_df)) > 1
                    else ["All"]
                ),
                key="pitcher_ar_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=1000,
                value=100,
                step=1,
                key="pitcher_ar_min_value",
            )
            filter_type = st.selectbox(
                "Filter By", ["IP", "TBF"], index=1, key="pitcher_ar_filter_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitchers_reg_df, "pitching_code"),
                index=0,
                key="pitcher_ar_team",
            )
            player_options, player_name_map = player_id_options(
                pitchers_reg_df, "pitcher_mlbid", "name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                key="pitcher_ar_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitchers_reg_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitchers_reg_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            if filter_type == "IP":
                df = numeric_filter(df, "IP", min_value)
            else:
                df = numeric_filter(df, "TBF", min_value)

            columns = [
                "name",
                "season",
                "pitching_code",
                "TBF",
                "IP",
                "stuff",
                "fastball_velo_reg",
                "max_velo_reg",
                "fastball_vaa_reg",
                "FA_pct_reg",
                "BB_rpm_reg",
                "SwStr_reg",
                "Ball_pct_reg",
                "Z_Contact_reg",
                "Chase_reg",
                "CSW_reg",
                "LA_lte_0_reg",
                "rel_z_reg",
                "rel_x_reg",
                "ext_reg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitching_code": "Team",
                "season": "Season",
                "stuff": "Pitch Grade",
                "fastball_velo_reg": "FA mph",
                "max_velo_reg": "Max FA mph",
                "fastball_vaa_reg": "FA VAA",
                "FA_pct_reg": "FA Usage (%)",
                "BB_rpm_reg": "BB Spin",
                "SwStr_reg": "SwStr (%)",
                "Ball_pct_reg": "Ball (%)",
                "Z_Contact_reg": "Z-Contact (%)",
                "Chase_reg": "Chase (%)",
                "CSW_reg": "CSW (%)",
                "LA_lte_0_reg": "LA<=0%",
                "rel_z_reg": "Vertical Release (ft.)",
                "rel_x_reg": "Horizontal Release (ft.)",
                "ext_reg": "Extension (ft.)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "pitchers_ar", "pitchers_ar_download")


def pitcher_splits():
    """Pitchers - Splits page (placeholder)"""
    st.title("Pitcher Splits")

    if pitcher_splits_df.empty:
        st.info("Missing pitcher_splits.csv")
        return

    tabs = st.tabs(["vL / vR", "Home / Away", "1H / 2H", "Monthly"])
    split_map = {
        "vL / vR": "vs L/R",
        "Home / Away": "Home/Away",
        "1H / 2H": "1st Half/2nd Half",
        "Monthly": "Monthly",
    }

    for idx, tab_name in enumerate(split_map.keys()):
        with tabs[idx]:
            split_type = split_map[tab_name]
            split_df = pitcher_splits_df[
                pitcher_splits_df["split_type"] == split_type
            ].copy()
            if split_df.empty:
                available = sorted(
                    pitcher_splits_df["split_type"].dropna().unique().tolist()
                )
                st.info(f"No data for {tab_name}. Available split types: {available}")
                continue

            left, right = st.columns([1, 3])
            with left:
                level = st.selectbox(
                    "Select Level",
                    ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                    index=1,
                    key=f"pitcher_splits_level_{idx}",
                )
                season = st.multiselect(
                    "Select Season",
                    season_options(split_df),
                    default=(
                        [season_options(split_df)[1]]
                        if len(season_options(split_df)) > 1
                        else ["All"]
                    ),
                    key=f"pitcher_splits_season_{idx}",
                )
                min_value = st.number_input(
                    "Minimum Value",
                    min_value=0,
                    max_value=1000,
                    value=100,
                    step=1,
                    key=f"pitcher_splits_min_value_{idx}",
                )
                filter_type = st.selectbox(
                    "Filter By",
                    ["IP", "TBF"],
                    index=1,
                    key=f"pitcher_splits_filter_type_{idx}",
                )
                split_choice = st.multiselect(
                    "Select Split",
                    ["All"] + sorted(split_df["split"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"pitcher_splits_split_{idx}",
                )
                team = st.selectbox(
                    "Select Team",
                    team_options(split_df, "pitching_code"),
                    index=0,
                    key=f"pitcher_splits_team_{idx}",
                )
                player_options, player_name_map = player_id_options(
                    split_df, "pitcher_mlbid", "name"
                )
                player = st.multiselect(
                    "Select Player",
                    player_options,
                    default=["All"],
                    format_func=lambda v: "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
                    key=f"pitcher_splits_player_{idx}",
                )
            with right:
                level_map = {
                    "All": [1, 11, 14, 16],
                    "MLB": [1],
                    "Triple-A": [11],
                    "Low-A": [14],
                    "Low Minors": [16],
                }
                base_stats = split_df.copy()
                base_stats = base_stats.assign(
                    __season=base_stats["season"],
                    __level=base_stats["level_id"],
                )
                df = split_df.copy()
                df = df[df["level_id"].isin(level_map[level])]
                df = filter_by_values(df, "season", season)
                df = filter_by_values(df, "split", split_choice)
                df = filter_by_team_token(df, "pitching_code", team)
                df = filter_by_values(df, "pitcher_mlbid", player)
                df = df.assign(__season=df["season"], __level=df["level_id"])

                if filter_type == "IP":
                    df = numeric_filter(df, "IP", min_value)
                else:
                    df = numeric_filter(df, "TBF", min_value)

                columns = [
                    "name",
                    "season",
                    "split",
                    "pitching_code",
                    "TBF",
                    "IP",
                    "stuff",
                    "fastball_velo",
                    "max_velo",
                    "fastball_vaa",
                    "FA_pct",
                    "BB_rpm",
                    "SwStr",
                    "Ball_pct",
                    "Z_Contact",
                    "Chase",
                    "CSW",
                    "LA_lte_0",
                    "rel_z",
                    "rel_x",
                    "ext",
                    "__season",
                    "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                if "BB_rpm" in df.columns:
                    df["BB_rpm"] = df["BB_rpm"].round(0)
                if "stuff" in df.columns:
                    df["stuff"] = df["stuff"].round(0)
                rename_map = {
                    "name": "Name",
                    "pitching_code": "Team",
                    "season": "Season",
                    "split": "Split",
                    "stuff": "Pitch Grade",
                    "fastball_velo": "FA mph",
                    "max_velo": "Max FA mph",
                    "fastball_vaa": "FA VAA",
                    "FA_pct": "FA Usage (%)",
                    "BB_rpm": "BB Spin",
                    "SwStr": "SwStr (%)",
                    "Ball_pct": "Ball (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "CSW": "CSW (%)",
                    "LA_lte_0": "LA<=0%",
                    "rel_z": "Vertical Release (ft.)",
                    "rel_x": "Horizontal Release (ft.)",
                    "ext": "Extension (ft.)",
                }
                df = df.rename(columns=rename_map)
                df = df.sort_values(by="Pitch Grade", ascending=False)
                stats_df = base_stats[
                    [col for col in columns if col in base_stats.columns]
                ].rename(columns=rename_map)
                render_table(
                    df,
                    reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                )
                download_button(
                    df,
                    f"pitcher_splits_{idx}",
                    f"pitcher_splits_download_{idx}",
                )


# =============================================================================
# INDIVIDUAL PITCHES PAGES
# =============================================================================


def pitch_shapes_outcomes():
    """Individual Pitches - Shapes and Outcomes page"""
    st.title("Individual Pitches - Shapes and Outcomes")

    if pitch_types.empty:
        st.info("Missing new_pitch_types.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitch_shapes_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitch_types),
                default=(
                    [season_options(pitch_types)[1]]
                    if len(season_options(pitch_types)) > 1
                    else ["All"]
                ),
                key="pitch_shapes_season",
            )
            min_pitches = st.number_input(
                "Minimum # Pitches",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                key="pitch_shapes_min_pitches",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitch_types, "pitching_code"),
                index=0,
                key="pitch_shapes_team",
            )
            pitcher_options, pitcher_name_map = player_id_options(
                pitch_types, "pitcher_mlbid", "name"
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                pitcher_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})",
                key="pitch_shapes_pitcher",
            )
            pitch_group = st.multiselect(
                "Select Pitch Group",
                ["All"] + sorted(pitch_types["pitch_group"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_shapes_pitch_group",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"] + sorted(pitch_types["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_shapes_pitch_tag",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitch_types.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitch_types.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_group", pitch_group)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "name",
                "pitching_code",
                "season",
                "pitch_tag",
                "pitches",
                "pct",
                "stuff",
                "velo",
                "max_velo",
                "vaa",
                "haa",
                "vbreak",
                "hbreak",
                "SwStr",
                "LA_lte_0",
                "Z_Contact",
                "Ball_pct",
                "Zone",
                "Chase",
                "CSW",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round stuff to integer
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            rename_map = {
                "name": "Name",
                "pitching_code": "Team",
                "season": "Season",
                "pitch_tag": "Pitch Type",
                "pitches": "#",
                "pct": "Usage (%)",
                "stuff": "Pitch Grade",
                "velo": "Velo",
                "max_velo": "Max Velo",
                "vaa": "VAA",
                "haa": "HAA",
                "vbreak": "IVB (in.)",
                "hbreak": "HB (in.)",
                "CSW": "CSW (%)",
                "SwStr": "SwStr (%)",
                "LA_lte_0": "LA<=0%",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "Zone": "Zone (%)",
                "Ball_pct": "Ball (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
            )
            download_button(df, "pitch_types", "pitch_types_download")


def pitch_ar():
    """Individual Pitches - Auto Regressed page"""
    st.title("Individual Pitches - Auto Regressed")

    if pitch_types_reg_df.empty:
        st.info("Missing pitch_types_regressed.csv or new_pitch_types.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitch_ar_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitch_types_reg_df),
                default=(
                    [season_options(pitch_types_reg_df)[1]]
                    if len(season_options(pitch_types_reg_df)) > 1
                    else ["All"]
                ),
                key="pitch_ar_season",
            )
            min_pitches = st.number_input(
                "Minimum # Pitches",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                key="pitch_ar_min_pitches",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitch_types_reg_df, "pitching_code"),
                index=0,
                key="pitch_ar_team",
            )
            pitcher_options, pitcher_name_map = player_id_options(
                pitch_types_reg_df, "pitcher_mlbid", "name"
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                pitcher_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})",
                key="pitch_ar_pitcher",
            )
            pitch_group = st.multiselect(
                "Select Pitch Group",
                ["All"]
                + sorted(pitch_types_reg_df["pitch_group"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_ar_pitch_group",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"]
                + sorted(pitch_types_reg_df["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_ar_pitch_tag",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitch_types_reg_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitch_types_reg_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_group", pitch_group)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "name",
                "pitching_code",
                "season",
                "pitch_tag",
                "pitches",
                "pct",
                "stuff",
                "velo_reg",
                "max_velo_reg",
                "vaa_reg",
                "haa_reg",
                "vbreak_reg",
                "hbreak_reg",
                "SwStr_reg",
                "LA_lte_0_reg",
                "Z_Contact_reg",
                "Ball_pct_reg",
                "Chase_reg",
                "CSW_reg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitching_code": "Team",
                "season": "Season",
                "pitch_tag": "Pitch Type",
                "pitches": "#",
                "pct": "Usage (%)",
                "stuff": "Pitch Grade",
                "velo_reg": "Velo",
                "max_velo_reg": "Max Velo",
                "vaa_reg": "VAA",
                "haa_reg": "HAA",
                "vbreak_reg": "IVB (in.)",
                "hbreak_reg": "HB (in.)",
                "CSW_reg": "CSW (%)",
                "SwStr_reg": "SwStr (%)",
                "LA_lte_0_reg": "LA<=0%",
                "Z_Contact_reg": "Z-Contact (%)",
                "Chase_reg": "Chase (%)",
                "Ball_pct_reg": "Ball (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
            )
            download_button(df, "pitch_types_ar", "pitch_types_ar_download")


def pitch_percentiles():
    """Individual Pitches - Percentiles page"""
    st.title("Individual Pitches - Percentiles")

    if pitch_types_pct.empty:
        st.info("Missing pitch_types_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitch_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitch_types_pct),
                default=(
                    [season_options(pitch_types_pct)[1]]
                    if len(season_options(pitch_types_pct)) > 1
                    else ["All"]
                ),
                key="pitch_pct_season",
            )
            min_pitches = st.number_input(
                "Minimum # Pitches",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                key="pitch_pct_min_pitches",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitch_types_pct, "pitching_code"),
                index=0,
                key="pitch_pct_team",
            )
            pitcher_options, pitcher_name_map = player_id_options(
                pitch_types_pct, "pitcher_mlbid", "name"
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                pitcher_options,
                default=["All"],
                format_func=lambda v: "All"
                if v == "All"
                else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})",
                key="pitch_pct_pitcher",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"]
                + sorted(pitch_types_pct["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_pct_pitch_tag",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = pitch_types_pct.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]

            columns = [
                "name",
                "pitching_code",
                "season",
                "pitch_tag",
                "pct",
                "stuff_z",
                "stuff_pctile",
                "velo_pctile",
                "max_velo_pctile",
                "vaa_pctile",
                "haa_pctile",
                "vbreak_pctile",
                "hbreak_pctile",
                "SwStr_pctile",
                "LA_lte_0_pctile",
                "Ball_pct_pctile",
                "Z_Contact_pctile",
                "Chase_pctile",
                "CSW_pctile",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitching_code": "Team",
                "season": "Season",
                "pitch_tag": "Pitch Type",
                "pct": "Usage (%)",
                "stuff_z": "Pitch Grade Z",
                "stuff_pctile": "Pitch Grade Pctile",
                "velo_pctile": "Velo",
                "max_velo_pctile": "Max Velo",
                "vaa_pctile": "VAA",
                "haa_pctile": "HAA",
                "vbreak_pctile": "IVB (in.)",
                "hbreak_pctile": "HB (in.)",
                "CSW_pctile": "CSW (%)",
                "SwStr_pctile": "SwStr (%)",
                "LA_lte_0_pctile": "LA<=0%",
                "Z_Contact_pctile": "Z-Contact (%)",
                "Chase_pctile": "Chase (%)",
                "Ball_pct_pctile": "Ball (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            render_table(
                df,
                reverse_cols={"VAA", "Ball (%)", "Z-Contact (%)"},
                label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
            )
            download_button(df, "pitch_percentiles", "pitch_pct_download")


def pitch_comps():
    """Individual Pitches - Pitch Level Comps page (placeholder)"""
    st.title("Pitch Level Comparisons")

    st.info("Pitch-level comparison functionality coming soon!")
    st.write("This will allow you to find similar pitches based on shape and outcomes.")


def pitch_splits():
    """Individual Pitches - Splits page (placeholder)"""
    st.title("Individual Pitch Splits")

    if pitch_type_splits_df.empty:
        st.info("Missing pitch_types_splits.csv")
        return

    tabs = st.tabs(["vL / vR", "Home / Away", "1H / 2H", "Monthly"])
    split_map = {
        "vL / vR": "vs L/R",
        "Home / Away": "Home/Away",
        "1H / 2H": "1st Half/2nd Half",
        "Monthly": "Monthly",
    }

    for idx, tab_name in enumerate(split_map.keys()):
        with tabs[idx]:
            split_type = split_map[tab_name]
            split_df = pitch_type_splits_df[
                pitch_type_splits_df["split_type"] == split_type
            ].copy()
            if split_df.empty:
                available = sorted(
                    pitch_type_splits_df["split_type"].dropna().unique().tolist()
                )
                st.info(f"No data for {tab_name}. Available split types: {available}")
                continue

            left, right = st.columns([1, 3])
            with left:
                level = st.selectbox(
                    "Select Level",
                    ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                    index=1,
                    key=f"pitch_splits_level_{idx}",
                )
                season = st.multiselect(
                    "Select Season",
                    season_options(split_df),
                    default=(
                        [season_options(split_df)[1]]
                        if len(season_options(split_df)) > 1
                        else ["All"]
                    ),
                    key=f"pitch_splits_season_{idx}",
                )
                min_pitches = st.number_input(
                    "Minimum # Pitches",
                    min_value=0,
                    max_value=1000,
                    value=50,
                    step=1,
                    key=f"pitch_splits_min_pitches_{idx}",
                )
                split_choice = st.multiselect(
                    "Select Split",
                    ["All"] + sorted(split_df["split"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"pitch_splits_split_{idx}",
                )
                team = st.selectbox(
                    "Select Team",
                    team_options(split_df, "pitching_code"),
                    index=0,
                    key=f"pitch_splits_team_{idx}",
                )
                pitcher_options, pitcher_name_map = player_id_options(
                    split_df, "pitcher_mlbid", "name"
                )
                pitcher = st.multiselect(
                    "Select Pitcher",
                    pitcher_options,
                    default=["All"],
                    format_func=lambda v: "All"
                    if v == "All"
                    else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})",
                    key=f"pitch_splits_pitcher_{idx}",
                )
                pitch_group = st.multiselect(
                    "Select Pitch Group",
                    (
                        ["All"]
                        + sorted(split_df["pitch_group"].dropna().unique().tolist())
                        if "pitch_group" in split_df.columns
                        else ["All"]
                    ),
                    default=["All"],
                    key=f"pitch_splits_pitch_group_{idx}",
                )
                pitch_tag = st.multiselect(
                    "Select Pitch Type",
                    ["All"] + sorted(split_df["pitch_tag"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"pitch_splits_pitch_tag_{idx}",
                )
            with right:
                level_map = {
                    "All": [1, 11, 14, 16],
                    "MLB": [1],
                    "Triple-A": [11],
                    "Low-A": [14],
                    "Low Minors": [16],
                }
                base_stats = split_df.copy()
                base_stats = base_stats.assign(
                    __season=base_stats["season"],
                    __level=base_stats["level_id"],
                )
                df = split_df.copy()
                df = df[df["level_id"].isin(level_map[level])]
                df = filter_by_values(df, "season", season)
                df = filter_by_values(df, "split", split_choice)
                df = filter_by_team_token(df, "pitching_code", team)
                df = filter_by_values(df, "pitcher_mlbid", pitcher)
                if "pitch_group" in df.columns:
                    df = filter_by_values(df, "pitch_group", pitch_group)
                df = filter_by_values(df, "pitch_tag", pitch_tag)
                df = df[df["pitches"] >= min_pitches]
                df = df.assign(__season=df["season"], __level=df["level_id"])

                columns = [
                    "name",
                    "pitching_code",
                    "season",
                    "split",
                    "pitch_tag",
                    "pitches",
                    "pct",
                    "stuff",
                    "velo",
                    "max_velo",
                    "vaa",
                    "haa",
                    "vbreak",
                    "hbreak",
                    "SwStr",
                    "Z_Contact",
                    "Ball_pct",
                    "Zone",
                    "Chase",
                    "CSW",
                    "__season",
                    "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                if "stuff" in df.columns:
                    df["stuff"] = df["stuff"].round(0)
                rename_map = {
                    "name": "Name",
                    "pitching_code": "Team",
                    "season": "Season",
                    "split": "Split",
                    "pitch_tag": "Pitch Type",
                    "pitches": "#",
                    "pct": "Usage (%)",
                    "stuff": "Pitch Grade",
                    "velo": "Velo",
                    "max_velo": "Max Velo",
                    "vaa": "VAA",
                    "haa": "HAA",
                    "vbreak": "IVB (in.)",
                    "hbreak": "HB (in.)",
                    "CSW": "CSW (%)",
                    "SwStr": "SwStr (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "Zone": "Zone (%)",
                    "Ball_pct": "Ball (%)",
                }
                df = df.rename(columns=rename_map)
                df = df.sort_values(by="Pitch Grade", ascending=False)
                stats_df = base_stats[
                    [col for col in columns if col in base_stats.columns]
                ].rename(columns=rename_map)
                render_table(
                    df,
                    reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                    label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
                )
                download_button(
                    df,
                    f"pitch_splits_{idx}",
                    f"pitch_splits_download_{idx}",
                )


# =============================================================================
# TEAMS PAGES
# =============================================================================


def team_hitting():
    """Team Hitting page"""
    st.title("Team Hitting")

    if team_damage.empty:
        st.info("Missing new_team_damage.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["MLB", "Triple-A", "Low-A", "Low Minors"],
                index=0,
                key="team_hitting_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(team_damage),
                default=(
                    [season_options(team_damage)[1]]
                    if len(season_options(team_damage)) > 1
                    else ["All"]
                ),
                key="team_hitting_season",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = team_damage.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = team_damage.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "hitting_code",
                "season",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "pull_FB_pct",
                "LA_gte_20",
                "LA_lte_0",
                "SEAGER",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "whiffs_vs_95",
                "contact_vs_avg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitting_code": "Team",
                "season": "Season",
                "bbe": "BBE",
                "damage_rate": "Damage/BBE (%)",
                "EV90th": "90th Pctile EV",
                "pull_FB_pct": "Pulled FB (%)",
                "LA_gte_20": "LA>=20%",
                "LA_lte_0": "LA<=0%",
                "selection_skill": "Selectivity (%)",
                "hittable_pitches_taken": "Hittable Pitch Take (%)",
                "chase": "Chase (%)",
                "z_con": "Z-Contact (%)",
                "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                "whiffs_vs_95": "Whiff vs. 95+ (%)",
                "contact_vs_avg": "Contact Over Expected (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols=HIGHER_IS_WORSE_COLS | {"Chase (%)", "LA<=0%"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                include_team_label=True,
            )
            download_button(df, "team_hitting", "team_hitting_download")


def team_pitching():
    """Team Pitching page"""
    st.title("Team Pitching")

    if team_stuff.empty:
        st.info("Missing new_team_stuff.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["MLB", "Triple-A", "Low-A", "Low Minors"],
                index=0,
                key="team_pitching_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(team_stuff),
                default=(
                    [season_options(team_stuff)[1]]
                    if len(season_options(team_stuff)) > 1
                    else ["All"]
                ),
                key="team_pitching_season",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = team_stuff.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = team_stuff.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "pitching_code",
                "season",
                "IP",
                "stuff",
                "fastball_velo",
                "fastball_vaa",
                "FA_pct",
                "SwStr",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "CSW",
                "LA_lte_0",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "pitching_code": "Team",
                "season": "Season",
                "stuff": "Pitch Grade",
                "fastball_velo": "FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "SwStr": "SwStr (%)",
                "Ball_pct": "Ball (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "CSW": "CSW (%)",
                "LA_lte_0": "LA<=0%",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                include_team_label=True,
            )
            download_button(df, "team_pitching", "team_pitching_download")


# =============================================================================
# LEAGUE PAGES
# =============================================================================


def league_hitting():
    """League - Hitting Stats page"""
    st.title("League Averages - Hitting")

    if hitting_avg.empty:
        st.info("Missing new_hitting_lg_avg.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            season = st.multiselect(
                "Select Season",
                season_options(hitting_avg),
                default=(
                    [season_options(hitting_avg)[1]]
                    if len(season_options(hitting_avg)) > 1
                    else ["All"]
                ),
                key="lg_hit_season",
            )
        with right:
            df = hitting_avg.copy()
            df = filter_by_values(df, "season", season)
            df = df.assign(
                Level=df["level_id"].map(
                    {1: "MLB", 11: "Triple-A", 14: "Low-A", 16: "Low Minors"}
                )
            )
            base_stats = hitting_avg.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = df.assign(__season=df["season"], __level=df["level_id"])
            columns = [
                "Level",
                "season",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "pull_FB_pct",
                "LA_gte_20",
                "LA_lte_0",
                "SEAGER",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "whiffs_vs_95",
                "contact_vs_avg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "season": "Season",
                "bbe": "BBE",
                "damage_rate": "Damage/BBE (%)",
                "EV90th": "90th Pctile EV",
                "pull_FB_pct": "Pulled FB (%)",
                "LA_gte_20": "LA>=20%",
                "LA_lte_0": "LA<=0%",
                "selection_skill": "Selectivity (%)",
                "hittable_pitches_taken": "Hittable Pitch Take (%)",
                "chase": "Chase (%)",
                "z_con": "Z-Contact (%)",
                "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                "whiffs_vs_95": "Whiff vs. 95+ (%)",
                "contact_vs_avg": "Contact Over Expected (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "league_hitting", "league_hitting_download")


def league_pitching():
    """League - Pitching Stats page"""
    st.title("League Averages - Pitching")

    if pitching_avg.empty:
        st.info("Missing new_lg_stuff.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["MLB", "Triple-A", "Low-A", "Low Minors"],
                index=0,
                key="lg_pitch_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitching_avg),
                default=(
                    [season_options(pitching_avg)[1]]
                    if len(season_options(pitching_avg)) > 1
                    else ["All"]
                ),
                key="lg_pitch_season",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitching_avg.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitching_avg.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "season",
                "stuff",
                "stuff_z",
                "fastball_velo",
                "fastball_vaa",
                "FA_pct",
                "BB_rpm",
                "SwStr",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "CSW",
                "LA_lte_0",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round BB_rpm and stuff to integers
            if "BB_rpm" in df.columns:
                df["BB_rpm"] = df["BB_rpm"].round(0)
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            rename_map = {
                "season": "Season",
                "CSW": "CSW (%)",
                "Ball_pct": "Ball (%)",
                "SwStr": "SwStr (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "fastball_velo": "FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "BB_rpm": "BB Spin",
                "stuff": "Pitch Grade",
                "stuff_z": "Pitch Grade Z",
                "LA_lte_0": "LA<=0%",
            }
            df = df.rename(columns=rename_map)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "league_pitching", "league_pitching_download")


def league_pitch_level():
    """League - Pitch Level Shapes and Outcomes page"""
    st.title("League Averages - Pitch Level Shapes and Outcomes")

    if league_pitch_types.empty:
        st.info("Missing league_pitch_types.csv")
        return
    if "throws" not in league_pitch_types.columns:
        st.info("league_pitch_types.csv is outdated. Please re-run data_aggregate.py.")
        return

    left, right = st.columns([1, 3])
    with left:
        season = st.multiselect(
            "Select Season",
            season_options(league_pitch_types),
            default=(
                [season_options(league_pitch_types)[1]]
                if len(season_options(league_pitch_types)) > 1
                else ["All"]
            ),
            key="lg_pitch_types_season",
        )
        throws = st.multiselect(
            "Select Throws",
            ["All"] + sorted(league_pitch_types["throws"].dropna().unique().tolist()),
            default=["All"],
            key="lg_pitch_types_throws",
        )
        pitch_tag = st.multiselect(
            "Select Pitch Type",
            ["All"]
            + sorted(league_pitch_types["pitch_tag"].dropna().unique().tolist()),
            default=["All"],
            key="lg_pitch_types_pitch_tag",
        )
    with right:
        base_stats = league_pitch_types.copy()
        base_stats = base_stats.assign(__season=base_stats["season"])
        df = league_pitch_types.copy()
        df = filter_by_values(df, "season", season)
        df = filter_by_values(df, "throws", throws)
        df = filter_by_values(df, "pitch_tag", pitch_tag)
        df = df.assign(__season=df["season"])

        columns = [
            "season",
            "throws",
            "pitch_tag",
            "pct",
            "velo",
            "vaa",
            "haa",
            "vbreak",
            "hbreak",
            "SwStr",
            "LA_lte_0",
            "Z_Contact",
            "Zone",
            "Ball_pct",
            "Chase",
            "CSW",
            "__season",
        ]
        df = df[[col for col in columns if col in df.columns]].copy()
        rename_map = {
            "season": "Season",
            "throws": "Throws",
            "pitch_tag": "Pitch Type",
            "pct": "Usage (%)",
            "velo": "Velo",
            "vaa": "VAA",
            "haa": "HAA",
            "vbreak": "IVB (in.)",
            "hbreak": "HB (in.)",
            "SwStr": "SwStr (%)",
            "LA_lte_0": "LA<=0%",
            "Z_Contact": "Z-Contact (%)",
            "Zone": "Zone (%)",
            "Ball_pct": "Ball (%)",
            "Chase": "Chase (%)",
            "CSW": "CSW (%)",
        }
        df = df.rename(columns=rename_map)
        stats_df = base_stats[
            [col for col in columns if col in base_stats.columns]
        ].rename(columns=rename_map)
        render_table(
            df,
            reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
            group_cols=["__season"],
            stats_df=stats_df,
        )
        download_button(df, "league_pitch_types", "league_pitch_types_download")


# =============================================================================
# GLOSSARY PAGES
# =============================================================================


def glossary_hitting():
    """Glossary - Hitting page"""
    st.title("Glossary - Hitting")

    st.markdown(
        """
### Hitting Metrics Glossary

**Damage/BBE (%)**: Percentage of batted ball events that result in "damage" (extra-base hits or hard-hit balls likely to result in positive outcomes).

**90th Pctile EV**: The 90th percentile exit velocity for a player's batted balls.

**Pulled FB (%)**: Percentage of fly balls that are pulled to the pull side.

**LA>=20%**: Percentage of batted balls with launch angle of 20 degrees or higher (fly balls).

**LA<=0%**: Percentage of batted balls with launch angle of 0 degrees or lower (ground balls).

**SEAGER**: A composite metric measuring overall hitting quality and approach.

**Selectivity (%)**: Measure of a hitter's ability to swing at strikes and take balls.

**Hittable Pitch Take (%)**: Percentage of hittable pitches that the batter takes (does not swing at).

**Chase (%)**: Percentage of pitches outside the zone that the batter swings at.

**Z-Contact (%)**: Contact rate on pitches in the strike zone.

**Whiff vs. Secondaries (%)**: Whiff rate against secondary pitches (breaking balls, offspeed).

**Whiff vs. 95+ (%)**: Whiff rate against fastballs 95 mph or higher.

**Contact Over Expected (%)**: Contact rate compared to expected contact rate based on pitch characteristics.
"""
    )


def glossary_pitching():
    """Glossary - Pitching page"""
    st.title("Glossary - Pitching")

    st.markdown(
        """
### Pitching Metrics Glossary

**Pitch Grade**: Overall pitch quality metric. Higher is better.

**FA mph**: Average fastball velocity.

**Max FA mph**: Maximum fastball velocity.

**FA VAA**: Fastball vertical approach angle.

**FA Usage (%)**: Percentage of pitches that are fastballs.

**BB Spin**: Baseball Savant spin rate (RPM).

**SwStr (%)**: Swinging strike percentage.

**Ball (%)**: Percentage of pitches resulting in balls.

**Z-Contact (%)**: Contact rate on pitches in the strike zone.

**Chase (%)**: Percentage of pitches outside the zone that induce swings.

**CSW (%)**: Called strikes plus whiffs percentage.

**LA<=0%**: Percentage of batted balls with launch angle of 0 degrees or lower (ground balls).

**Vertical Release (ft.)**: Vertical release point in feet.

**Horizontal Release (ft.)**: Horizontal release point in feet.

**Extension (ft.)**: Release point extension toward home plate in feet.

**VAA**: Vertical approach angle (for individual pitches).

**HAA**: Horizontal approach angle (for individual pitches).

**IVB (in.)**: Induced vertical break in inches.

**HB (in.)**: Horizontal break in inches.

**Zone (%)**: Percentage of pitches thrown in the strike zone.
"""
    )


# =============================================================================
# NAVIGATION SETUP
# =============================================================================

# Step 1: Check if user is logged in
try:
    is_logged_in = st.user.is_logged_in
except AttributeError:
    is_logged_in = False

if not is_logged_in:
    st.subheader("🔐 Login Required")
    st.markdown(
        """
        Please log in to access the premium features of this app.
        """
    )
    if st.button("Log in with Google", type="primary"):
        st.login()
    st.stop()

# Step 2: User is logged in, now check subscription
st.markdown(f"Welcome back, **{st.user.name}**! 👋")
st.markdown("---")

st.subheader("Premium Access Required")
st.markdown(
    """
To access all features and data in this app, please subscribe below.
Your subscription supports ongoing development and maintenance of these analytics tools.
"""
)

# Check subscription status - this will stop execution if user is not subscribed
add_auth(
    required=True,
    show_redirect_button=True,
    subscription_button_text="Subscribe to Access Premium Features",
    button_color="#FF4B4B",
)

# Only subscribed users will see content below this point
st.success("✅ You have premium access! Enjoy all features.")
st.markdown("---")

# Define page navigation with hierarchical groups
pages = {
    "Home": [
        st.Page(home_page, title="Welcome", icon="🏠"),
    ],
    "Hitters": [
        st.Page(hitter_individual_stats, title="Individual Stats", icon="⚾"),
        st.Page(hitter_percentiles, title="Percentiles", icon="📊"),
        st.Page(hitter_comps, title="Hitter Comps", icon="🔍"),
        st.Page(hitter_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(hitter_splits, title="Splits", icon="📋"),
    ],
    "Pitchers": [
        st.Page(pitcher_individual_stats, title="Individual Stats", icon="⚾"),
        st.Page(pitcher_percentiles, title="Percentiles", icon="📊"),
        st.Page(pitcher_comps, title="Pitcher Comps", icon="🔍"),
        st.Page(pitcher_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(pitcher_splits, title="Splits", icon="📋"),
    ],
    "Individual Pitches": [
        st.Page(pitch_shapes_outcomes, title="Shapes and Outcomes", icon="🎯"),
        st.Page(pitch_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(pitch_percentiles, title="Percentiles", icon="📊"),
        st.Page(pitch_comps, title="Pitch Level Comps", icon="🔍"),
        st.Page(pitch_splits, title="Splits", icon="📋"),
    ],
    "Teams": [
        st.Page(team_hitting, title="Team Hitting", icon="🏆"),
        st.Page(team_pitching, title="Team Pitching", icon="🏆"),
    ],
    "League": [
        st.Page(league_hitting, title="Hitting Stats", icon="🌐"),
        st.Page(league_pitching, title="Pitching Stats", icon="🌐"),
        st.Page(league_pitch_level, title="Pitch Level Shapes", icon="🌐"),
    ],
    "Glossary": [
        st.Page(glossary_hitting, title="Hitting Glossary", icon="📖"),
        st.Page(glossary_pitching, title="Pitching Glossary", icon="📖"),
    ],
}

# Create and run navigation
pg = st.navigation(pages)
pg.run()
