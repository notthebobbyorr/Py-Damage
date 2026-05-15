from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import colors

from app.auth import _is_user_subscribed
from app.config import DEFAULT_NO_FORMAT_COLS, PREVIEW_ROWS
from app.filters import apply_column_filters

_TABLE_COUNTER = 0


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
        if col not in numeric_cols
        and pd.api.types.is_object_dtype(plot_df[col])
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
        st.plotly_chart(fig, width="stretch", key=f"{table_key}_plot")


def render_table(
    df: pd.DataFrame,
    reverse_cols: set[str] | None = None,
    no_format_cols: set[str] | None = None,
    group_cols: list[str] | None = None,
    stats_df: pd.DataFrame | None = None,
    abs_cols: set[str] | None = None,
    show_controls: bool = True,
    include_team_label: bool = False,
    label_cols: list[str] | None = None,
    hide_cols: set[str] | None = None,
    round_decimals: int = 1,
    default_sort_col: str | None = None,
    fixed_scale_cols: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    if df.empty:
        st.info("No data available yet.")
        return
    if not _is_user_subscribed():
        st.info(
            f"Preview mode: showing the first {PREVIEW_ROWS} rows. Subscribe for full access."
        )
        df = df.head(PREVIEW_ROWS)
        show_controls = False

    global _TABLE_COUNTER
    table_key = f"table_{_TABLE_COUNTER}"
    _TABLE_COUNTER += 1

    if show_controls:
        df = apply_column_filters(df, table_key)
        if df.empty:
            st.info("No data after filters.")
            return

    def _contains_non_mlb_rows(table_df: pd.DataFrame) -> bool:
        level_cols = ["__level", "level_id", "Level"]
        for col in level_cols:
            if col not in table_df.columns:
                continue
            s = table_df[col]
            if col in {"__level", "level_id"}:
                vals = pd.to_numeric(s, errors="coerce")
                if vals.notna().any() and (vals != 1).any():
                    return True
            else:
                text = s.astype(str).str.strip().str.lower()
                milb_tokens = {"triple-a", "low-a", "low minors"}
                if text.isin(milb_tokens).any():
                    return True
        return False

    hide_cols = hide_cols or set()
    # Hide Player ID by default from display while keeping it for downloads.
    hide_cols = set(hide_cols) | {"Player ID"}
    # Hide Team column whenever the displayed table contains any non-MLB rows.
    if _contains_non_mlb_rows(df):
        hide_cols = set(hide_cols) | {"Team"}
    display_cols = [
        col for col in df.columns if not col.startswith("__") and col not in hide_cols
    ]
    df_display = df[display_cols].copy()

    _render_plot_controls(
        df_display,
        table_key,
        include_team_label,
        reverse_cols or set(),
        label_cols,
    )

    if show_controls:
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 1, 1, 1])
        with ctrl_col1:
            sort_options = list(df_display.columns)
            sort_index = (
                sort_options.index(default_sort_col)
                if default_sort_col in sort_options
                else 0
            )
            sort_col = st.selectbox(
                "Sort by",
                options=sort_options,
                index=sort_index,
                key=f"{table_key}_sort_col",
            )
        with ctrl_col2:
            sort_dir = st.selectbox(
                "Order",
                options=["Desc", "Asc"],
                key=f"{table_key}_sort_dir",
            )
        with ctrl_col3:
            page_size_option = st.selectbox(
                "Rows per page",
                options=["All", 25, 50, 100, 200],
                index=2,
                key=f"{table_key}_page_size",
            )
        ascending = sort_dir == "Asc"
        df_display = df_display.sort_values(sort_col, ascending=ascending, na_position="last")
        df = df.loc[df_display.index]

        total_rows = len(df_display)
        if page_size_option == "All":
            page_size = total_rows
            page = 1
        else:
            page_size = int(page_size_option)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            with ctrl_col4:
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
    abs_cols = abs_cols or set()
    no_format_cols = no_format_cols or DEFAULT_NO_FORMAT_COLS
    numeric_cols = df_display.select_dtypes(include="number").columns
    float_cols = df.select_dtypes(include="floating").columns
    format_cols = [col for col in numeric_cols if col not in no_format_cols]

    if len(numeric_cols) > 0:
        df_page_display[numeric_cols] = df_page_display[numeric_cols].round(
            round_decimals
        )
    if len(float_cols) > 0:
        df_page_display[float_cols] = df_page_display[float_cols].round(round_decimals)

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
        abs_format_cols = [col for col in abs_cols if col in stats_source.columns]
        abs_stats_source = stats_source.copy()
        if abs_format_cols:
            abs_stats_source[abs_format_cols] = abs_stats_source[abs_format_cols].abs()
        if group_cols:
            if stats_format_cols:
                q10 = stats_source.groupby(group_cols)[stats_format_cols].quantile(0.05)
                q90 = stats_source.groupby(group_cols)[stats_format_cols].quantile(0.95)
                med = stats_source.groupby(group_cols)[stats_format_cols].median()
            else:
                q10 = q90 = med = None
            if abs_format_cols:
                q10_abs = abs_stats_source.groupby(group_cols)[
                    abs_format_cols
                ].quantile(0.05)
                q90_abs = abs_stats_source.groupby(group_cols)[
                    abs_format_cols
                ].quantile(0.95)
                med_abs = abs_stats_source.groupby(group_cols)[abs_format_cols].median()
            else:
                q10_abs = q90_abs = med_abs = None
        else:
            if stats_format_cols:
                q10 = stats_source[stats_format_cols].quantile(0.05)
                q90 = stats_source[stats_format_cols].quantile(0.95)
                med = stats_source[stats_format_cols].median()
            else:
                q10 = q90 = med = None
            if abs_format_cols:
                q10_abs = abs_stats_source[abs_format_cols].quantile(0.05)
                q90_abs = abs_stats_source[abs_format_cols].quantile(0.95)
                med_abs = abs_stats_source[abs_format_cols].median()
            else:
                q10_abs = q90_abs = med_abs = None
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
                    row_q10_abs = row_q90_abs = row_med_abs = None
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
                    if q10_abs is None:
                        row_q10_abs = row_q90_abs = row_med_abs = None
                    else:
                        row_q10_abs = q10_abs.loc[group_key]
                        row_q90_abs = q90_abs.loc[group_key]
                        row_med_abs = med_abs.loc[group_key]
            else:
                row_q10 = q10
                row_q90 = q90
                row_med = med
                row_q10_abs = q10_abs
                row_q90_abs = q90_abs
                row_med_abs = med_abs

            styles: list[str] = []
            for col in row.index:
                if col not in format_cols:
                    styles.append("")
                    continue
                if fixed_scale_cols and col in fixed_scale_cols:
                    vmin, vcenter, vmax = fixed_scale_cols[col]
                elif col in similarity_medians:
                    vmin = 0
                    vmax = 99
                    vcenter = similarity_medians[col]
                else:
                    if col not in stats_format_cols or row_q10 is None:
                        styles.append("")
                        continue
                    if col in abs_cols and row_q10_abs is not None:
                        vmin = row_q10_abs[col]
                        vmax = row_q90_abs[col]
                        vcenter = row_med_abs[col]
                    else:
                        vmin = row_q10[col]
                        vmax = row_q90[col]
                        vcenter = row_med[col]
                if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
                    styles.append("")
                    continue
                if pd.isna(vcenter):
                    styles.append("")
                    continue
                # TwoSlopeNorm requires strict ordering: vmin < vcenter < vmax.
                if not (vmin < vcenter < vmax):
                    vcenter = (vmin + vmax) / 2
                if not (vmin < vcenter < vmax):
                    styles.append("")
                    continue
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                val = row[col]
                if pd.isna(val):
                    styles.append("")
                    continue
                if col in abs_cols:
                    val = abs(val)
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
            format_map = {col: f"{{:.{round_decimals}f}}" for col in float_cols}
            # Format integer-value columns without decimals
            int_substr_kws = ["Similarity", "Pitch Grade", "BB Spin", "#"]
            for col in df_page_display.columns:
                if any(kw in col for kw in int_substr_kws) or col.endswith("Pctile"):
                    format_map[col] = "{:.0f}"
            styler = styler.format(format_map)
        st.dataframe(styler, width="stretch", hide_index=True)
        return
    if len(float_cols) > 0:
        # Identify columns that should display as integers
        int_substr_kws = ["Similarity", "Pitch Grade", "BB Spin", "#"]
        int_cols = [
            col
            for col in df_page_display.columns
            if any(kw in col for kw in int_substr_kws) or col.endswith("Pctile")
        ]
        other_float_cols = [col for col in float_cols if col not in int_cols]

        if other_float_cols:
            df_page_display[other_float_cols] = df_page_display[
                other_float_cols
            ].applymap(lambda x: f"{x:.{round_decimals}f}" if pd.notna(x) else x)
        for col in int_cols:
            if col in df_page_display.columns:
                df_page_display[col] = df_page_display[col].apply(
                    lambda x: f"{x:.0f}" if pd.notna(x) else x
                )
    st.dataframe(df_page_display, width="stretch", hide_index=True)
