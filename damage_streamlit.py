from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import colors
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
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_damage_df() -> pd.DataFrame:
    # Prefer the most comprehensive file with newest data
    preferred_files = [
        DATA_DIR / "damage_pos_2015_2025.csv",
    ]
    for preferred in preferred_files:
        if preferred.exists():
            return pd.read_csv(preferred)
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


def render_table(
    df: pd.DataFrame,
    reverse_cols: set[str] | None = None,
    no_format_cols: set[str] | None = None,
    group_cols: list[str] | None = None,
    stats_df: pd.DataFrame | None = None,
    show_controls: bool = True,
) -> None:
    if df.empty:
        st.info("No data available yet.")
        return

    if show_controls:
        global _TABLE_COUNTER
        table_key = f"table_{_TABLE_COUNTER}"
        _TABLE_COUNTER += 1

        df = apply_column_filters(df, table_key)
        if df.empty:
            st.info("No data after filters.")
            return

    display_cols = [col for col in df.columns if not col.startswith("__")]
    df_display = df[display_cols].copy()

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
pitcher_df = _normalize_team_col(pitcher_df, "pitching_code")
pitcher_df = _normalize_la_cols(pitcher_df)
pitcher_pct = _normalize_team_col(pitcher_pct, "pitching_code")
pitch_types = _normalize_team_col(pitch_types, "pitching_code")
pitch_types_pct = _normalize_team_col(pitch_types_pct, "pitching_code")
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
                default=[season_options(damage_df)[1]] if len(season_options(damage_df)) > 1 else ["All"],
                key="hitter_stats_season",
            )
            min_value = st.number_input(
                "Minimum Value", min_value=0, max_value=500, value=100, step=1,
                key="hitter_stats_min_value",
            )
            value_type = st.selectbox("Filter By", ["PA", "BBE"], index=1, key="hitter_stats_value_type")
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(damage_df["hitting_code"].dropna().unique().tolist()),
                default=["All"],
                key="hitter_stats_team",
            )
            player = st.multiselect(
                "Select Player",
                ["All"] + sorted(damage_df["hitter_name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "hitting_code", team)
            df = filter_by_values(df, "hitter_name", player)
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
                default=[season_options(hitter_pct)[1]] if len(season_options(hitter_pct)) > 1 else ["All"],
                key="hitter_pct_season",
            )
            min_value = st.number_input(
                "Minimum Value", min_value=0, max_value=500, value=100, step=1,
                key="hitter_pct_min_value",
            )
            value_type = st.selectbox("Filter By", ["PA", "BBE"], index=1, key="hitter_pct_value_type")
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(hitter_pct["hitting_code"].dropna().unique().tolist()),
                default=["All"],
                key="hitter_pct_team",
            )
            player = st.multiselect(
                "Select Player",
                ["All"] + sorted(hitter_pct["hitter_name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "hitting_code", team)
            df = filter_by_values(df, "hitter_name", player)

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
                reverse_cols={"Hittable Pitch Take", "Chase", "Whiff vs Secondaries", "Whiff vs 95+"},
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
            season_choice = st.selectbox("Season", seasons, index=0, key="hitter_comps_season")
            season_df = player_pool[player_pool["season"] == season_choice]
            players = sorted(season_df["hitter_name"].dropna().unique().tolist())
            player_choice = st.selectbox("Player", players, index=0, key="hitter_comps_player")
            player_df = season_df[season_df["hitter_name"] == player_choice]
            teams = sorted(player_df["hitting_code"].dropna().unique().tolist())
            team_choice = (
                st.selectbox("Team", teams, index=0, key="hitter_comps_team")
                if len(teams) > 1
                else (teams[0] if teams else None)
            )
            if team_choice:
                player_df = player_df[player_df["hitting_code"] == team_choice]

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
                    ~(eligible_comp["hitter_name"] == player_choice)
                ]
                stats = eligible_comp[feature_cols]
                means = stats.mean()
                stds = stats.std(ddof=0).replace(0, np.nan)
                zscores = (stats - means) / stds
                zscores = zscores.fillna(0)
                target_vec = (
                    ((player_df[feature_cols] - means) / stds).fillna(0).iloc[0].to_numpy()
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
                default=[season_options(hitters_reg_df)[1]] if len(season_options(hitters_reg_df)) > 1 else ["All"],
                key="hitter_ar_season",
            )
            min_value = st.number_input(
                "Minimum Value", min_value=0, max_value=500, value=100, step=1,
                key="hitter_ar_min_value",
            )
            value_type = st.selectbox("Filter By", ["PA", "BBE"], index=1, key="hitter_ar_value_type")
            team = st.multiselect(
                "Select Team",
                ["All"]
                + sorted(hitters_reg_df["hitting_code"].dropna().unique().tolist()),
                default=["All"],
                key="hitter_ar_team",
            )
            player = st.multiselect(
                "Select Player",
                ["All"]
                + sorted(hitters_reg_df["hitter_name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "hitting_code", team)
            df = filter_by_values(df, "hitter_name", player)

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

    st.info("Hitter splits functionality coming soon!")

    tabs = st.tabs(["vL / vR", "Home / Away", "1H / 2H", "Monthly"])

    with tabs[0]:
        st.subheader("vs. Lefty / vs. Righty")
        st.write("Split data by pitcher handedness will be available here.")

    with tabs[1]:
        st.subheader("Home / Away")
        st.write("Home and away split data will be available here.")

    with tabs[2]:
        st.subheader("1st Half / 2nd Half")
        st.write("First half and second half split data will be available here.")

    with tabs[3]:
        st.subheader("Monthly Splits")
        st.write("Monthly split data will be available here.")


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
                default=[season_options(pitcher_df)[1]] if len(season_options(pitcher_df)) > 1 else ["All"],
                key="pitcher_stats_season",
            )
            min_value = st.number_input(
                "Minimum Value", min_value=0, max_value=1000, value=100, step=1,
                key="pitcher_stats_min_value",
            )
            filter_type = st.selectbox(
                "Filter By", ["IP", "TBF"], index=1, key="pitcher_stats_filter_type"
            )
            team = st.multiselect(
                "Select Team",
                ["All"]
                + sorted(pitcher_df["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pitcher_stats_team",
            )
            player = st.multiselect(
                "Select Player",
                ["All"] + sorted(pitcher_df["name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", player)
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
                default=[season_options(pitcher_pct)[1]] if len(season_options(pitcher_pct)) > 1 else ["All"],
                key="pitcher_pct_season",
            )
            min_value = st.number_input(
                "Minimum TBF", min_value=0, max_value=1000, value=100, step=1,
                key="pitcher_pct_min_value",
            )
            team = st.multiselect(
                "Select Team",
                ["All"]
                + sorted(pitcher_pct["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pitcher_pct_team",
            )
            player = st.multiselect(
                "Select Player",
                ["All"] + sorted(pitcher_pct["name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", player)
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
    st.title("Pitcher Comparisons")

    if pitcher_df.empty:
        st.info("Missing pitcher_stuff_new.csv")
    else:
        player_pool = pitcher_df.copy()
        player_pool = player_pool[
            (player_pool["level_id"] == 1) & (player_pool["IP"] >= 5)
        ]
        eligible_all = pitcher_df.copy()
        eligible_all = eligible_all[
            (eligible_all["level_id"] == 1) & (eligible_all["IP"] >= 50)
        ]
        if player_pool.empty:
            st.info("No eligible MLB pitcher seasons (min 5 IP).")
        else:
            seasons = season_options(player_pool, "season")[1:]
            season_choice = st.selectbox("Season", seasons, index=0, key="pitcher_comps_season")
            season_df = player_pool[player_pool["season"] == season_choice]
            players = sorted(season_df["name"].dropna().unique().tolist())
            player_choice = st.selectbox("Player", players, index=0, key="pitcher_comps_player")
            player_df = season_df[season_df["name"] == player_choice]
            teams = sorted(player_df["pitching_code"].dropna().unique().tolist())
            team_choice = (
                st.selectbox("Team", teams, index=0, key="pitcher_comps_team")
                if len(teams) > 1
                else (teams[0] if teams else None)
            )
            if team_choice:
                player_df = player_df[player_df["pitching_code"] == team_choice]

            feature_cols = [
                "stuff",
                "fastball_velo",
                "fastball_vaa",
                "FA_pct",
                "BB_rpm",
                "SwStr",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "LA_lte_0",
                "rel_z",
                "rel_x",
                "ext",
            ]
            feature_cols = [c for c in feature_cols if c in eligible_all.columns]
            eligible_comp = eligible_all.dropna(subset=feature_cols)
            if player_df.empty:
                st.info("No season row found for that selection.")
            else:
                eligible_comp = eligible_comp[
                    ~(eligible_comp["name"] == player_choice)
                ]
                stats = eligible_comp[feature_cols]
                means = stats.mean()
                stds = stats.std(ddof=0).replace(0, np.nan)
                zscores = (stats - means) / stds
                zscores = zscores.fillna(0)
                target_vec = (
                    ((player_df[feature_cols] - means) / stds).fillna(0).iloc[0].to_numpy()
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
                df = eligible_comp[[col for col in display_cols if col in eligible_comp.columns]].copy()
                df = df.rename(
                    columns={
                        "name": "Name",
                        "pitching_code": "Team",
                        "season": "Season",
                        "fastball_velo": "FA mph",
                        "fastball_vaa": "FA VAA",
                        "SwStr": "SwStr (%)",
                        "Ball_pct": "Ball (%)",
                        "Chase": "Chase (%)",
                        "Z_Contact": "Z-Contact (%)",
                        "LA_lte_0": "LA<=0%",
                        "rel_z": "Vertical Release (ft.)",
                        "rel_x": "Horizontal Release (ft.)",
                        "ext": "Extension (ft.)",
                        "similarity_score": "Similarity (0-100)",
                        "stuff": "Pitch Grade",
                        "stuff_z": "Pitch Grade Z",
                        "FA_pct": "FA Usage (%)",
                        "BB_rpm": "BB Spin",
                    }
                )
                stats_df = pitcher_df.copy()
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
                    "fastball_velo",
                    "fastball_vaa",
                    "FA_pct",
                    "BB_rpm",
                    "SwStr",
                    "Ball_pct",
                    "Z_Contact",
                    "Chase",
                    "LA_lte_0",
                    "rel_z",
                    "rel_x",
                    "ext",
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
                        "fastball_velo": "FA mph",
                        "fastball_vaa": "FA VAA",
                        "SwStr": "SwStr (%)",
                        "Ball_pct": "Ball (%)",
                        "Chase": "Chase (%)",
                        "Z_Contact": "Z-Contact (%)",
                        "LA_lte_0": "LA<=0%",
                        "rel_z": "Vertical Release (ft.)",
                        "rel_x": "Horizontal Release (ft.)",
                        "ext": "Extension (ft.)",
                        "stuff": "Pitch Grade",
                        "FA_pct": "FA Usage (%)",
                        "BB_rpm": "BB Spin",
                    }
                )
                target_display_cols = [
                    "name",
                    "pitching_code",
                    "season",
                    "TBF",
                    "IP",
                    "stuff",
                    "fastball_velo",
                    "fastball_vaa",
                    "FA_pct",
                    "BB_rpm",
                    "SwStr",
                    "Ball_pct",
                    "Z_Contact",
                    "Chase",
                    "LA_lte_0",
                    "rel_z",
                    "rel_x",
                    "ext",
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
                        "fastball_velo": "FA mph",
                        "fastball_vaa": "FA VAA",
                        "SwStr": "SwStr (%)",
                        "Ball_pct": "Ball (%)",
                        "Chase": "Chase (%)",
                        "Z_Contact": "Z-Contact (%)",
                        "LA_lte_0": "LA<=0%",
                        "rel_z": "Vertical Release (ft.)",
                        "rel_x": "Horizontal Release (ft.)",
                        "ext": "Extension (ft.)",
                        "stuff": "Pitch Grade",
                        "FA_pct": "FA Usage (%)",
                        "BB_rpm": "BB Spin",
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
                default=[season_options(pitchers_reg_df)[1]] if len(season_options(pitchers_reg_df)) > 1 else ["All"],
                key="pitcher_ar_season",
            )
            min_value = st.number_input(
                "Minimum Value", min_value=0, max_value=1000, value=100, step=1,
                key="pitcher_ar_min_value",
            )
            filter_type = st.selectbox(
                "Filter By", ["IP", "TBF"], index=1, key="pitcher_ar_filter_type"
            )
            team = st.multiselect(
                "Select Team",
                ["All"]
                + sorted(pitchers_reg_df["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pitcher_ar_team",
            )
            player = st.multiselect(
                "Select Player",
                ["All"]
                + sorted(pitchers_reg_df["name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", player)
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

    st.info("Pitcher splits functionality coming soon!")

    tabs = st.tabs(["vL / vR", "Home / Away", "1H / 2H", "Monthly"])

    with tabs[0]:
        st.subheader("vs. Lefty / vs. Righty")
        st.write("Split data by batter handedness will be available here.")

    with tabs[1]:
        st.subheader("Home / Away")
        st.write("Home and away split data will be available here.")

    with tabs[2]:
        st.subheader("1st Half / 2nd Half")
        st.write("First half and second half split data will be available here.")

    with tabs[3]:
        st.subheader("Monthly Splits")
        st.write("Monthly split data will be available here.")


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
                default=[season_options(pitch_types)[1]] if len(season_options(pitch_types)) > 1 else ["All"],
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
            team = st.multiselect(
                "Select Team",
                ["All"]
                + sorted(pitch_types["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_shapes_team",
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                ["All"] + sorted(pitch_types["name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", pitcher)
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
                default=[season_options(pitch_types_reg_df)[1]] if len(season_options(pitch_types_reg_df)) > 1 else ["All"],
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
            team = st.multiselect(
                "Select Team",
                ["All"]
                + sorted(pitch_types_reg_df["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_ar_team",
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                ["All"] + sorted(pitch_types_reg_df["name"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_ar_pitcher",
            )
            pitch_group = st.multiselect(
                "Select Pitch Group",
                ["All"] + sorted(pitch_types_reg_df["pitch_group"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_ar_pitch_group",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"] + sorted(pitch_types_reg_df["pitch_tag"].dropna().unique().tolist()),
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
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", pitcher)
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
                default=[season_options(pitch_types_pct)[1]] if len(season_options(pitch_types_pct)) > 1 else ["All"],
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
            team = st.multiselect(
                "Select Team",
                ["All"]
                + sorted(
                    pitch_types_pct["pitching_code"].dropna().unique().tolist()
                ),
                default=["All"],
                key="pitch_pct_team",
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                ["All"]
                + sorted(pitch_types_pct["name"].dropna().unique().tolist()),
                default=["All"],
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
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", pitcher)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]

            columns = [
                "name",
                "pitching_code",
                "season",
                "pitch_tag",
                "pct",
                "pct_pctile",
                "stuff_z",
                "stuff_pctile",
                "velo_pctile",
                "max_velo_pctile",
                "vaa_pctile",
                "haa_pctile",
                "vbreak_pctile",
                "hbreak_pctile",
                "SwStr_pctile",
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
                "pct_pctile": "Usage (%) Pctile",
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
                "Z_Contact_pctile": "Z-Contact (%)",
                "Chase_pctile": "Chase (%)",
                "Ball_pct_pctile": "Ball (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            render_table(df, reverse_cols={"VAA", "Ball (%)", "Z-Contact (%)"})
            download_button(df, "pitch_percentiles", "pitch_pct_download")


def pitch_comps():
    """Individual Pitches - Pitch Level Comps page (placeholder)"""
    st.title("Pitch Level Comparisons")

    st.info("Pitch-level comparison functionality coming soon!")
    st.write("This will allow you to find similar pitches based on shape and outcomes.")


def pitch_splits():
    """Individual Pitches - Splits page (placeholder)"""
    st.title("Individual Pitch Splits")

    st.info("Individual pitch splits functionality coming soon!")

    tabs = st.tabs(["vL / vR", "Home / Away", "1H / 2H", "Monthly"])

    with tabs[0]:
        st.subheader("vs. Lefty / vs. Righty")
        st.write("Split data by batter handedness will be available here.")

    with tabs[1]:
        st.subheader("Home / Away")
        st.write("Home and away split data will be available here.")

    with tabs[2]:
        st.subheader("1st Half / 2nd Half")
        st.write("First half and second half split data will be available here.")

    with tabs[3]:
        st.subheader("Monthly Splits")
        st.write("Monthly split data will be available here.")


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
                default=[season_options(team_damage)[1]] if len(season_options(team_damage)) > 1 else ["All"],
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
                default=[season_options(team_stuff)[1]] if len(season_options(team_stuff)) > 1 else ["All"],
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
                default=[season_options(hitting_avg)[1]] if len(season_options(hitting_avg)) > 1 else ["All"],
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
                default=[season_options(pitching_avg)[1]] if len(season_options(pitching_avg)) > 1 else ["All"],
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
    """League - Pitch Level Shapes and Outcomes page (placeholder)"""
    st.title("League Averages - Pitch Level Shapes and Outcomes")

    st.info("League-level pitch shapes and outcomes functionality coming soon!")
    st.write("This will show league averages for different pitch types across levels and seasons.")


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

# Check if user is logged in and subscribed
st.title("Profiles")

st.markdown(
    """
Welcome! Here you will find metrics I (https://twitter.com/NotTheBobbyOrr) have developed for analyzing hitters & pitchers at a player and team level.
I make frequent use of these statistics in my work at BaseballProspectus dot com (https://www.baseballprospectus.com/author/ringtheodubel/) and for my own fantasy strategy.
"""
)

st.markdown("---")

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
