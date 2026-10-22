from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import colors

from app.auth import _is_user_subscribed
from app.config import DEFAULT_NO_FORMAT_COLS, PREVIEW_ROWS
from app.filters import apply_column_filters
from app.table_layout import RAW_COUNT_COLUMNS, order_stat_columns

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
    if not any([player_col, team_col, pitch_col, split_col]) and not label_cols:
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
    resolved_label_cols = list(dict.fromkeys(resolved_label_cols))

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
    plot_label_count: int | None = None,
) -> None:
    from app.plot_variables import plot_values, MONTH_ORDER, is_month_variable, default_plot_type
    plot_df, numeric_cols = _coerce_numeric_for_plot(df, exclude_cols=set(label_cols or []))
    variables = [c for c in plot_df if not c.startswith("__")]
    if len(variables) < 2:
        return
    with st.expander("Create-a-Plot", expanded=False):
        chart_type = st.selectbox("Plot type",["Scatter","Line"],key=f"{table_key}_plot_kind")
        col1, col2 = st.columns(2)
        x_col = col1.selectbox(
            "X column",
            options=variables,
            index=variables.index(numeric_cols[0]) if numeric_cols else 0,
            key=f"{table_key}_plot_x",
        )
        y_options = [col for col in variables if col != x_col]
        if not y_options:
            y_options = variables
        y_default = next((y_options.index(c) for c in numeric_cols if c in y_options), 0)
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
        color_options = ["(none)"] + variables
        color_col = col4.selectbox(
            "Color",
            options=color_options,
            index=0,
            key=f"{table_key}_plot_color",
        )
        modes = {}
        roles = [("X",x_col),("Y",y_col)] + ([] if size_col == "(none)" else [("Size",size_col)]) + ([] if color_col == "(none)" else [("Color",color_col)])
        type_controls = st.columns(len(roles))
        def sync_type(field_key,widget_key):
            st.session_state[field_key] = st.session_state[widget_key]
        for control, (role, variable) in zip(type_controls,roles):
            choices = ["Time", "Continuous", "Discrete / factor"]
            default = default_plot_type(plot_df[variable],variable)
            field_key = f"{table_key}_plot_type_{variable}"
            widget_key = f"{table_key}_plot_{role.lower()}_type_{variable}"
            st.session_state[widget_key] = st.session_state.get(field_key,default)
            modes[variable] = control.selectbox(f"{role} type",choices,key=widget_key,on_change=sync_type,args=(field_key,widget_key))
        discrete = [c for c,mode in modes.items() if mode == "Discrete / factor"]
        st.caption("Time uses a calendar axis. Continuous uses numeric values and color gradients. Discrete / factor uses categories and distinct colors; month categories follow calendar order.")
        if plot_label_count is None:
            max_points = st.number_input(
                "Max labeled points (sampled if exceeded)",
                min_value=100,
                max_value=20000,
                value=100,
                step=100,
                key=f"{table_key}_plot_max",
            )
        else:
            max_points = plot_label_count
        show_labels = st.checkbox(
            "Show point labels",
            value=True,
            key=f"{table_key}_plot_labels",
        )
        label_fields = []
        if show_labels:
            label_variables = list(dict.fromkeys(variable for _,variable in roles))
            include_variables = st.checkbox("Include selected plot variables",value=False,key=f"{table_key}_label_variables")
            if include_variables:
                label_fields.extend(st.multiselect("Plot variables in labels",label_variables,default=label_variables,key=f"{table_key}_label_variable_fields"))
            include_metadata = st.checkbox("Include customary metadata",value=True,key=f"{table_key}_label_metadata")
            if include_metadata:
                candidates = ["Name","Player","Season","Team","Level","Pitch Type","Pitcher Hand","Opponent Hand",
                              "Split","Split Type","Home / Away","Half","Month","Pitch Group","Date"]
                metadata = [c for c in candidates if c in plot_df]
                defaults = [c for c in ["Name","Player","Season","Team","Pitch Type"] if c in metadata]
                requested = {c.lower() for c in (label_cols or [])}
                defaults.extend(c for c in metadata if c.lower() in requested and c not in defaults)
                label_fields.extend(st.multiselect("Metadata in labels",metadata,default=defaults,key=f"{table_key}_label_metadata_fields"))
            label_fields = list(dict.fromkeys(label_fields))
        if plot_label_count is not None:
            st.caption(f"Labels up to {plot_label_count} points; all points remain plotted.")
        else:
            st.caption("Labels highlight statistical extremes up to the selected maximum.")

        plot_df = plot_df.copy()
        label_source = plot_df.reset_index(drop=True)

        size_arg = None if size_col == "(none)" else size_col
        color_arg = None if color_col == "(none)" else color_col
        if size_arg and modes.get(size_arg,"Continuous") != "Continuous":
            st.info("Size requires a continuous numeric variable.")
            return
        colorscale = None
        color_midpoint = None
        time_labels = {c: plot_df[c].astype(str).reset_index(drop=True) for c in [x_col,y_col]}
        time_color_ticks = None
        try:
            for variable in dict.fromkeys([x_col, y_col, color_arg, size_arg]):
                if variable:
                    plot_df[variable] = plot_values(plot_df[variable], modes.get(variable,"Continuous"), variable, years=plot_df.get("Season"))
        except ValueError as exc:
            st.info(str(exc))
            return
        if color_arg is not None and color_arg not in discrete:
            if pd.api.types.is_datetime64_any_dtype(plot_df[color_arg]):
                dates = plot_df[color_arg].dropna().drop_duplicates().sort_values()
                dates = dates.iloc[np.linspace(0,len(dates)-1,min(6,len(dates)),dtype=int)] if len(dates) else dates
                time_color_ticks = ([value.timestamp() for value in dates], dates.dt.strftime("%Y-%m-%d").tolist())
                if is_month_variable(df[color_col],color_col):
                    time_color_ticks = (time_color_ticks[0],df.loc[dates.index,color_col].astype(str).tolist())
                plot_df["__time_color"] = plot_df[color_arg].map(lambda value: value.timestamp() if pd.notna(value) else np.nan)
                color_arg = "__time_color"
            if not pd.api.types.is_numeric_dtype(plot_df[color_arg]):
                st.info("Continuous color requires numeric values.")
                return
            colorscale = "RdYlGn_r" if color_arg in reverse_cols else "RdYlGn"
            color_midpoint = float(plot_df[color_arg].median())
        plot_df = plot_df.reset_index(drop=True).dropna(subset=[x_col, y_col])
        if size_arg:
            plot_df = plot_df[plot_df[size_arg].notna() & (plot_df[size_arg] >= 0)]
        if plot_df.empty:
            st.info("No rows available for the selected variables.")
            return
        if x_col in plot_df.columns and y_col in plot_df.columns:
            extremes = pd.Series(False, index=plot_df.index)
            continuous_axes = [c for c in [x_col,y_col] if c not in discrete]
            for axis in continuous_axes:
                extremes |= (plot_df[axis] <= plot_df[axis].quantile(0.1)) | (plot_df[axis] >= plot_df[axis].quantile(0.9))
            if not continuous_axes:
                extremes[:] = True
            if show_labels and label_fields:
                labels = _build_point_labels(
                    label_source.loc[plot_df.index],
                    include_team=False,
                    label_cols=label_fields,
                )
                if labels is not None:
                    label_mask = pd.Series(True, index=plot_df.index) if plot_label_count is not None else extremes.copy()
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
            category_orders={c: [month for month in MONTH_ORDER if month in set(df[c].dropna().astype(str))] for c in discrete if is_month_variable(df[c],c)},
        )
        if show_labels and "__label" in plot_df.columns:
            fig.update_traces(textposition="top center", mode="markers+text")
        else:
            fig.update_traces(mode="markers")
        fig.update_xaxes(type="category" if x_col in discrete else "date" if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]) else "linear")
        fig.update_yaxes(type="category" if y_col in discrete else "date" if pd.api.types.is_datetime64_any_dtype(plot_df[y_col]) else "linear")
        for axis, variable in [("x",x_col),("y",y_col)]:
            if modes[variable] == "Time" and is_month_variable(df[variable],variable):
                ticks = pd.DataFrame({"value":plot_df[variable],"label":time_labels[variable].loc[plot_df.index]}).dropna().drop_duplicates().sort_values("value")
                if ticks.value.dt.year.nunique() > 1:
                    ticks["label"] = ticks.label + " " + ticks.value.dt.year.astype(str)
                getattr(fig, f"update_{axis}axes")(tickvals=ticks.value.tolist(),ticktext=ticks.label.tolist())
        if time_color_ticks is not None:
            fig.update_coloraxes(colorbar=dict(title=color_col,tickvals=time_color_ticks[0],ticktext=time_color_ticks[1]))
        month_hover = {}
        for variable in [x_col,y_col]:
            if is_month_variable(df[variable],variable):
                month_hover[variable] = time_labels[variable].loc[plot_df.index].tolist()
        if month_hover:
            hover = pd.DataFrame(month_hover,index=plot_df.index)
            # Plotly traces may be partitioned by categorical color.
            for trace in fig.data:
                rows = plot_df if color_arg is None or color_arg not in discrete else plot_df[plot_df[color_arg].astype(str)==trace.name]
                trace.customdata = hover.loc[rows.index].values
                for index,variable in enumerate(month_hover):
                    axis = "x" if variable == x_col else "y"
                    trace.hovertemplate = trace.hovertemplate.replace(f"%{{{axis}}}",f"%{{customdata[{index}]}}")
        if chart_type == "Line":
            from app.plot_variables import add_plot_lines
            add_plot_lines(fig,plot_df,x_col,y_col,color_arg,color_arg in discrete,MONTH_ORDER)
        fig.update_traces(marker=dict(opacity=0.7, **({} if size_arg else {"size":7})))
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
    plot_label_count: int | None = None,
    conditional_formatting: bool = True,
    official_hitting_details: bool = False,
    official_pitching_details: bool = False,
    official_team_details: bool = False,
    workload_kind: str | None = None,
    show_maximum_filters: bool = False,
    damage_after_la: bool = False,
    pitch_outcomes_after_csw: bool = False,
) -> pd.DataFrame:
    if df.empty:
        st.info("No data available yet.")
        return df
    pitching = workload_kind == "pitcher" or official_pitching_details
    df = df[order_stat_columns(df.columns, pitching=bool(pitching), damage_after_la=damage_after_la,
                               pitch_outcomes_after_csw=pitch_outcomes_after_csw)]
    if not _is_user_subscribed():
        st.info(
            f"Preview mode: showing the first {PREVIEW_ROWS} rows. Subscribe for full access."
        )
        df = df.head(PREVIEW_ROWS)
        show_controls = False

    global _TABLE_COUNTER
    table_key = f"table_{_TABLE_COUNTER}"
    _TABLE_COUNTER += 1

    from app.filters import apply_maximum_filters
    if show_maximum_filters:
        df = apply_maximum_filters(df, table_key, workload_kind)
    if df.empty:
        st.info("No data after maximum filters.")
        return df

    if show_controls:
        df = apply_column_filters(df, table_key)
        if df.empty:
            st.info("No data after filters.")
            return df

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
    reverse_cols = set(reverse_cols or ())
    if workload_kind == "pitcher" or official_pitching_details:
        reverse_cols.discard("Avg Miss Distance (in.)")
    else:
        reverse_cols.add("Avg Miss Distance (in.)")
    for column in ['Whiff (%)', 'Whiff%', 'SwStr (%)', 'SwStr%']:
        if pitching or column in (fixed_scale_cols or {}):
            reverse_cols.discard(column)
        else:
            reverse_cols.add(column)

    _render_plot_controls(
        df_display,
        table_key,
        include_team_label,
        reverse_cols or set(),
        label_cols,
        plot_label_count,
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

    def display_table(table):
        if official_hitting_details or official_pitching_details:
            from app.official_hitting import render_official_table
            render_official_table(table, df_page_full, table_key + "_official", pitching=official_pitching_details, team=official_team_details)
        else:
            st.dataframe(table, width="stretch", hide_index=True)

    max_elements = pd.get_option("styler.render.max_elements")
    total_cells = df_page_display.shape[0] * df_page_display.shape[1]
    reverse_cols = reverse_cols or set()
    abs_cols = abs_cols or set()
    no_format_cols = set(no_format_cols or DEFAULT_NO_FORMAT_COLS) | RAW_COUNT_COLUMNS
    numeric_cols = df_display.select_dtypes(include="number").columns
    float_cols = df_display.select_dtypes(include="floating").columns
    format_cols = [col for col in numeric_cols if col not in no_format_cols]

    if len(numeric_cols) > 0:
        df_page_display[numeric_cols] = df_page_display[numeric_cols].round(
            round_decimals
        )
    if len(float_cols) > 0:
        df_page_display[float_cols] = df_page_display[float_cols].round(round_decimals)

    if conditional_formatting and len(format_cols) > 0 and total_cells <= max_elements:
        stats_source = stats_df if stats_df is not None else df
        similarity_cols = [col for col in format_cols if col.startswith("Similarity")]
        stats_format_cols = [col for col in format_cols if col in stats_source.columns]
        if not stats_format_cols and not similarity_cols:
            display_table(df_page_display)
            return df
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
            int_substr_kws = ["Similarity", "Pitch Grade", "BR spin", "#"]
            for col in df_page_display.columns:
                if any(kw in col for kw in int_substr_kws) or col.endswith("Pctile"):
                    format_map[col] = "{:.0f}"
            styler = styler.format(format_map)
        display_table(styler)
        return df
    if len(float_cols) > 0:
        # Identify columns that should display as integers
        int_substr_kws = ["Similarity", "Pitch Grade", "BR spin", "#"]
        int_cols = [
            col
            for col in df_page_display.columns
            if any(kw in col for kw in int_substr_kws) or col.endswith("Pctile")
        ]
        other_float_cols = [col for col in float_cols if col not in int_cols]

        if other_float_cols:
            df_page_display[other_float_cols] = df_page_display[
                other_float_cols
            ].map(lambda x: f"{x:.{round_decimals}f}" if pd.notna(x) else x)
        for col in int_cols:
            if col in df_page_display.columns:
                df_page_display[col] = df_page_display[col].apply(
                    lambda x: f"{x:.0f}" if pd.notna(x) else x
                )
    display_table(df_page_display)
    return df
