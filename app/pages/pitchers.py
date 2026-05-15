from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.config import (
    ABS_GRADIENT_COLS_PITCHERS,
    GAME_TYPE_GROUP_NOTE,
    LEVEL_LABELS,
    PITCHER_COMPS_BASE_FEATURE_COLS,
    PITCHER_COMPS_EXTRA_FEATURE_COLS,
    PITCHER_REVERSE_DISPLAY_COLS,
)
from app.datasets import (
    get_pitcher_gamelogs,
    pitcher_df,
    pitcher_mlb_eq_coeffs,
    pitcher_pct,
    pitcher_splits_df,
    pitchers_mlb_eq_df,
    pitchers_reg_df,
)
from app.filters import (
    download_button,
    filter_by_game_type_group,
    filter_by_team_token,
    filter_by_values,
    game_type_group_options,
    pitcher_workload_filter,
    player_id_options,
    season_options,
    team_options,
)
from app.utils import (
    _pitcher_display_map,
    _similarity_choice_labels,
    constant_pctile_subset,
    maybe_add_level_col,
    rank_for_display,
)
from app.viz import render_table


_PITCHER_PCT_COLS = [
    "stuff", "grade_v13", "fastball_velo", "max_velo", "fastball_vaa",
    "SwStr", "Ball_pct", "Z_Contact", "Chase", "CSW",
    "rel_z", "rel_x", "ext",
    "p_SwStr_pct", "Damage_pct", "p_Damage_pct", "takeoff_rate",
]
_PITCHER_PCT_REVERSE = {
    "fastball_vaa", "Ball_pct", "Z_Contact", "takeoff_rate",
    "Damage_pct", "p_Damage_pct",
}


@st.cache_resource
def _pitcher_constant_pct() -> pd.DataFrame:
    """Constant percentile basis: Regular Season, TBF >= 150, per season + level."""
    return constant_pctile_subset(
        pitcher_pct,
        cols=_PITCHER_PCT_COLS,
        workload_col="TBF",
        workload_min=150,
        reverse_cols=_PITCHER_PCT_REVERSE,
    )


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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(pitcher_df),
                index=0,
                key="pitcher_stats_game_type_group",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                key="pitcher_stats_min_value",
            )
            filter_type = st.selectbox(
                "Filter By", ["IP", "TBF", "GS"], index=1, key="pitcher_stats_filter_type"
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
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
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
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
            df = pitcher_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            df = pitcher_workload_filter(df, filter_type, min_value)

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "TBF",
                "IP",
                "GS",
                "HR",
                "stuff",
                "grade_v13",
                "fastball_velo",
                "max_velo",
                "fastball_vaa",
                "FA_pct",
                "BB_rpm",
                "SwStr",
                "p_SwStr_pct",
                "Swing_pct",
                "p_Swing_pct",
                "Damage_pct",
                "p_Damage_pct",
                "Zone",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "CSW",
                "LA_lte_0",
                "rel_z",
                "rel_x",
                "ext",
                "inf_arm_angle",
                "SBO",
                "SB",
                "takeoff_rate",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round BB_rpm, stuff, and grade_v13 to integers
            if "BB_rpm" in df.columns:
                df["BB_rpm"] = df["BB_rpm"].round(0)
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            if "grade_v13" in df.columns:
                df["grade_v13"] = df["grade_v13"].round(0).astype("Int64")
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "GS": "GS",
                "HR": "HR",
                "stuff": "Pitch Grade",
                "grade_v13": "Execution Grade",
                "fastball_velo": "FA mph",
                "max_velo": "Max FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "BB_rpm": "BB Spin",
                "SwStr": "SwStr (%)",
                "p_SwStr_pct": "pSwStr (%)",
                "Swing_pct": "Swing (%)",
                "p_Swing_pct": "pSwing (%)",
                "Damage_pct": "Damage/BBE (%)",
                "p_Damage_pct": "pDamage/BBE (%)",
                "Zone": "Zone (%)",
                "Ball_pct": "Ball (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "CSW": "CSW (%)",
                "LA_lte_0": "LA<=0%",
                "rel_z": "Vertical Release (ft.)",
                "rel_x": "Horizontal Release (ft.)",
                "ext": "Extension (ft.)",
                "inf_arm_angle": "Inferred Arm Angle",
                "SBO": "SBO",
                "SB": "SB",
                "takeoff_rate": "Takeoff% Against",
            }
            df = df.rename(columns=rename_map)
            df = maybe_add_level_col(df, level)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)", "Damage/BBE (%)", "pDamage/BBE (%)", "HR", "SBO", "SB", "Takeoff% Against"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCHERS,
            )
            download_button(df, "pitchers", "pitchers_download")


def pitcher_percentiles():
    """Pitchers - Percentiles page"""
    st.title("Pitcher Percentiles")

    if pitcher_pct.empty:
        st.info("Missing pitcher_pctiles.csv")
    else:
        _LEVEL_MAP = {
            "All": [1, 11, 14, 16],
            "MLB": [1],
            "Triple-A": [11],
            "Low-A": [14],
            "Low Minors": [16],
        }
        left, right = st.columns([1, 3])
        with left:
            mode = st.radio(
                "Percentile Mode",
                ["Customizable", "Constant"],
                index=0,
                key="pitcher_pct_mode",
                help=(
                    "Customizable: percentiles recompute against the population "
                    "matching your filters and minimum threshold. "
                    "Constant: stable, season-level percentiles drawn from a fixed "
                    "Regular-Season, 150+ TBF population. Use Constant when filtering "
                    "to a single player so the displayed ranks reflect league-wide "
                    "context rather than the player's own row."
                ),
            )
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
            if mode == "Customizable":
                game_type_group = st.selectbox(
                    "Game Type",
                    game_type_group_options(pitcher_pct),
                    index=0,
                    key="pitcher_pct_game_type_group",
                )
                min_value = st.number_input(
                    "Minimum Value",
                    min_value=0,
                    max_value=2000,
                    value=20,
                    step=1,
                    key="pitcher_pct_min_value",
                )
                filter_type = st.selectbox(
                    "Filter By",
                    ["TBF", "IP", "GS"],
                    index=0,
                    key="pitcher_pct_filter_type",
                )
            else:
                game_type_group = "Regular Season"
                min_value = 0
                filter_type = "TBF"
                st.caption(
                    "Constant mode: Regular Season, TBF ≥ 150, ranked within each season + level. "
                    "Recommended when viewing a single player across years."
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
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitcher_pct_player",
            )
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
            if mode == "Constant":
                df = _pitcher_constant_pct().copy()
            else:
                df = pitcher_pct.copy()
            df = df[df["level_id"].isin(_LEVEL_MAP[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)

            if mode == "Customizable":
                df = pitcher_workload_filter(df, filter_type, min_value)

                df = rank_for_display(
                    df,
                    _PITCHER_PCT_COLS,
                    ["season", "level_id", "game_type_group"],
                    reverse_cols=_PITCHER_PCT_REVERSE,
                )

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "stuff_pctile",
                "grade_v13_pctile",
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
                "p_SwStr_pct_pctile",
                "Damage_pct_pctile",
                "p_Damage_pct_pctile",
                "takeoff_rate_pctile",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
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
                "grade_v13_pctile": "Execution Grade Pctile",
                "p_SwStr_pct_pctile": "pSwStr (%)",
                "Damage_pct_pctile": "Damage/BBE%",
                "p_Damage_pct_pctile": "pDamage/BBE%",
                "takeoff_rate_pctile": "Takeoff Against (%)",
            }
            df = df.rename(columns=rename_map)
            df = maybe_add_level_col(df, level)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            _pctile_scale = (1, 50, 100)
            _fixed = {
                col: _pctile_scale for col in [
                    "Pitch Grade Pctile", "Execution Grade Pctile",
                    "Avg FA mph", "Max FA mph", "FA VAA",
                    "SwStr (%)", "Ball (%)", "Z-Contact (%)", "Chase (%)", "CSW (%)",
                    "Vertical Release (ft.)", "Horizontal Release (ft.)", "Extension (ft.)",
                    "pSwStr (%)", "Damage/BBE%", "pDamage/BBE%", "Takeoff Against (%)",
                ]
            }
            render_table(
                df,
                abs_cols=ABS_GRADIENT_COLS_PITCHERS,
                round_decimals=0,
                fixed_scale_cols=_fixed,
            )
            download_button(df, "pitcher_percentiles", "pitcher_pct_download")


def pitcher_comps():
    """Pitchers - Comparisons page"""
    st.title("Pitcher Comparisons (Auto-Regressed)")

    if pitchers_reg_df.empty:
        st.info("Missing pitchers_regressed.csv or pitcher_stuff_new.csv")
        return

    use_mlb_eq = st.toggle(
        "Use MLB-equivalent translated stats",
        value=False,
        key="pitcher_comps_use_mlb_eq",
        help=(
            "Use intra+inter-season level translations to compare non-MLB seasons "
            "against MLB seasons in MLB-equivalent space."
        ),
    )
    comp_df = pitchers_reg_df.copy()
    if use_mlb_eq:
        if pitchers_mlb_eq_df.empty:
            st.info(
                "MLB-equivalent translation table is unavailable; using raw AR stats."
            )
            use_mlb_eq = False
        else:
            comp_df = pitchers_mlb_eq_df.copy()

    if "game_type_group" in comp_df.columns:
        comp_df = comp_df[comp_df["game_type_group"] != "Spring Training"]

    if use_mlb_eq:
        player_pool = comp_df[(comp_df["TBF"] >= 20)].copy()
        eligible_all = comp_df[
            (comp_df["level_id"] == 1) & (comp_df["TBF"] >= 200)
        ].copy()
        if player_pool.empty:
            st.info("No eligible pitcher seasons (min 20 TBF).")
            return
        target_levels = sorted(player_pool["level_id"].dropna().unique().tolist())
        if not target_levels:
            st.info("No target levels available.")
            return
        target_level = st.selectbox(
            "Target Level",
            target_levels,
            index=0,
            key="pitcher_comps_target_level",
            format_func=lambda v: LEVEL_LABELS.get(int(v), str(int(v))),
        )
        player_pool = player_pool[player_pool["level_id"] == target_level]
    else:
        player_pool = comp_df[(comp_df["level_id"] == 1) & (comp_df["IP"] >= 5)].copy()
        eligible_all = comp_df[
            (comp_df["level_id"] == 1) & (comp_df["IP"] >= 50)
        ].copy()

    if player_pool.empty:
        if use_mlb_eq:
            st.info("No eligible pitcher seasons for that level (min 20 TBF).")
        else:
            st.info("No eligible MLB pitcher seasons (min 5 IP).")
        return
    if eligible_all.empty:
        if use_mlb_eq:
            st.info("No eligible MLB comparison seasons (min 200 TBF).")
        else:
            st.info("No eligible MLB comparison seasons (min 50 IP).")
        return

    seasons = season_options(player_pool, "season")[1:]
    if not seasons:
        st.info("No seasons available for this view.")
        return
    season_choice = st.selectbox("Season", seasons, index=0, key="pitcher_comps_season")
    season_df = player_pool[player_pool["season"] == season_choice]
    if season_df.empty:
        st.info("No player rows for this season selection.")
        return

    player_options, player_name_map = player_id_options(
        season_df, "pitcher_mlbid", "name"
    )
    player_values = [opt for opt in player_options if opt != "All"]
    if not player_values:
        st.info("No players available for this filter.")
        return
    player_choice = st.selectbox(
        "Player",
        player_values,
        index=0,
        format_func=lambda v: f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
        key="pitcher_comps_player",
    )
    player_df = season_df[season_df["pitcher_mlbid"] == player_choice]
    teams = team_options(player_df, "pitching_code")
    team_choice = st.selectbox("Team", teams, index=0, key="pitcher_comps_team")
    if team_choice and team_choice != "All":
        filtered_player_df = filter_by_team_token(
            player_df, "pitching_code", team_choice
        )
        if filtered_player_df.empty:
            st.warning(
                "No rows for that team at this level/season; using all team rows for selection."
            )
        else:
            player_df = filtered_player_df

    metric_suffix = "_mlb_eq" if use_mlb_eq else ""
    default_feature_cols = [
        f"{col}{metric_suffix}" for col in PITCHER_COMPS_BASE_FEATURE_COLS
    ]
    allowed_cols = list(
        dict.fromkeys(
            default_feature_cols
            + [f"{col}{metric_suffix}" for col in PITCHER_COMPS_EXTRA_FEATURE_COLS]
        )
    )
    display_map = _pitcher_display_map(include_mlb_eq=False)
    if use_mlb_eq:
        base_display_map = _pitcher_display_map(include_mlb_eq=False)
        for col, label in base_display_map.items():
            if (
                col
                in PITCHER_COMPS_BASE_FEATURE_COLS + PITCHER_COMPS_EXTRA_FEATURE_COLS
            ):
                display_map[f"{col}_mlb_eq"] = label

    exclude_cols = {
        "batter_mlbid",
        "pitcher_mlbid",
        "level_id",
        "game_pk",
        "PA",
        "IP",
        "TBF",
        "GS",
        "pitches",
        "pitches_n",
        "pitches_num",
        "pitches_den",
        "bbe",
        "season",
    }
    numeric_cols, similarity_labels = _similarity_choice_labels(
        eligible_all, display_map, exclude_cols
    )
    numeric_cols = [col for col in numeric_cols if col in allowed_cols]
    default_feature_cols = [col for col in default_feature_cols if col in numeric_cols]
    similarity_key = (
        "pitcher_comps_similarity_cols_mlb_eq"
        if use_mlb_eq
        else "pitcher_comps_similarity_cols_raw"
    )
    feature_cols = st.multiselect(
        "Similarity Score Columns",
        options=numeric_cols,
        default=default_feature_cols,
        key=similarity_key,
        format_func=lambda col: similarity_labels.get(col, col),
    )
    feature_cols = [col for col in feature_cols if col in numeric_cols]
    feature_cols = list(dict.fromkeys(feature_cols))
    if not feature_cols:
        st.info("Select at least one column to compute similarity scores.")
        return
    if player_df.empty:
        st.info("No season row found for that selection.")
        return

    eligible_comp = eligible_all.copy()
    eligible_comp = eligible_comp[~(eligible_comp["pitcher_mlbid"] == player_choice)]
    eligible_comp = eligible_comp[eligible_comp[feature_cols].notna().any(axis=1)]
    if eligible_comp.empty:
        st.info("No comparable MLB rows found after filters.")
        return

    stats = eligible_comp[feature_cols].copy()
    means = stats.mean().fillna(0.0)
    stats = stats.fillna(means)
    stds = stats.std(ddof=0).replace(0, np.nan)
    zscores = ((stats - means) / stds).fillna(0)
    target_stats = player_df[feature_cols].copy().fillna(means)
    target_vec = ((target_stats - means) / stds).fillna(0).iloc[0].to_numpy()
    distances = np.linalg.norm(zscores.to_numpy() - target_vec, axis=1)
    max_dist = distances.max() if len(distances) else 0.0
    if max_dist == 0:
        similarity = np.full_like(distances, 100.0, dtype=float)
    else:
        similarity = 100 * (1 - (distances / max_dist))

    eligible_comp = eligible_comp.copy()
    eligible_comp["similarity_score"] = similarity.round(0)
    eligible_comp = eligible_comp.sort_values("similarity_score", ascending=False)
    eligible_comp = eligible_comp.assign(
        __season=eligible_comp["season"], __level=eligible_comp["level_id"]
    )

    base_rename = {
        "name": "Name",
        "pitching_code": "Team",
        "season": "Season",
        "similarity_score": "Similarity (0-100)",
    }

    display_cols = [
        "name",
        "pitching_code",
        "season",
        "TBF",
        "IP",
        "GS",
        "similarity_score",
        *feature_cols,
        "__season",
        "__level",
    ]
    df = eligible_comp[
        [col for col in display_cols if col in eligible_comp.columns]
    ].copy()
    if "grade_v13" in df.columns:
        df["grade_v13"] = df["grade_v13"].round(0).astype("Int64")
    df = df.rename(columns={**base_rename, **similarity_labels})
    df = df.loc[:, ~df.columns.duplicated()]

    stats_df = eligible_all.copy()
    stats_df = stats_df.assign(
        __season=stats_df["season"], __level=stats_df["level_id"]
    )
    stats_columns = [
        "name",
        "pitching_code",
        "season",
        "TBF",
        "IP",
        "GS",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season",
        "__level",
    ]
    stats_df = stats_df[
        [col for col in stats_columns if col in stats_df.columns]
    ].copy()
    if "grade_v13" in stats_df.columns:
        stats_df["grade_v13"] = stats_df["grade_v13"].round(0).astype("Int64")
    stats_df = stats_df.rename(columns={**base_rename, **similarity_labels})
    stats_df = stats_df.loc[:, ~stats_df.columns.duplicated()]

    target_cols = [
        "name",
        "pitching_code",
        "season",
        "TBF",
        "IP",
        "GS",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season",
        "__level",
    ]
    target_df = player_df.assign(
        __season=player_df["season"], __level=player_df["level_id"]
    )
    target_df = target_df[
        [col for col in target_cols if col in target_df.columns]
    ].copy()
    if "grade_v13" in target_df.columns:
        target_df["grade_v13"] = target_df["grade_v13"].round(0).astype("Int64")
    if use_mlb_eq and "__level" in target_df.columns:
        target_df["__level"] = 1
    target_df = target_df.rename(columns={**base_rename, **similarity_labels})
    target_df = target_df.loc[:, ~target_df.columns.duplicated()]

    if use_mlb_eq:
        level_label = LEVEL_LABELS.get(int(target_level), str(int(target_level)))
        player_name = str(
            player_df["name"].iloc[0]
            if "name" in player_df.columns and not player_df.empty
            else player_name_map.get(player_choice, "Player")
        )
        st.caption(
            f"{player_name}'s MLB-equivalent statistics derived from their "
            f"{level_label} statistics:"
        )
    else:
        st.caption("Selected season")
    render_table(
        target_df,
        reverse_cols=PITCHER_REVERSE_DISPLAY_COLS,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
        abs_cols=ABS_GRADIENT_COLS_PITCHERS,
        show_controls=False,
        hide_cols={"Team"},
    )
    if use_mlb_eq:
        st.caption(
            "Most similar MLB seasons by translated MLB-equivalent stats (TBF >= 200)"
        )
    else:
        st.caption("Most similar MLB seasons (IP >= 50)")
    render_table(
        df,
        reverse_cols=PITCHER_REVERSE_DISPLAY_COLS,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
        abs_cols=ABS_GRADIENT_COLS_PITCHERS,
        default_sort_col="Similarity (0-100)",
    )


def pitcher_mlb_equivalencies():
    """Pitchers - MLB equivalency translations (intra+inter-season chained)."""
    st.title("Pitcher MLB Equivalencies")

    if pitchers_mlb_eq_df.empty:
        st.info("MLB-equivalent table is unavailable.")
        return

    st.caption(
        "Intra+inter-season chained translations in season-adjusted z-score space "
        "(16->14->11->1)."
    )
    st.caption(
        "Pitch Grade, FA mph, Vertical Release, Horizontal Release, Extension, and "
        "Arm Angle are treated as MLB-equivalent pass-through metrics."
    )
    st.caption("Direct edge fits use TBF >= 60 at both source and destination levels.")

    view = pitchers_mlb_eq_df.copy()
    if "TBF" in view.columns:
        min_tbf = st.number_input(
            "Minimum TBF",
            min_value=0,
            max_value=2000,
            value=20,
            step=10,
            key="pitcher_mlb_eq_min_tbf",
        )
        view = view[view["TBF"] >= min_tbf]

    season_vals = season_options(view)
    season = st.selectbox(
        "Season",
        season_vals,
        index=(1 if len(season_vals) > 1 else 0),
        key="pitcher_mlb_eq_season",
    )
    view = filter_by_values(view, "season", season)

    level_options = ["All", "MLB", "Triple-A", "Low-A", "Low Minors"]
    level_choice = st.selectbox(
        "Level",
        level_options,
        index=0,
        key="pitcher_mlb_eq_level",
    )
    level_map = {
        "All": [1, 11, 14, 16],
        "MLB": [1],
        "Triple-A": [11],
        "Low-A": [14],
        "Low Minors": [16],
    }
    view = view[view["level_id"].isin(level_map[level_choice])]

    team = st.selectbox(
        "Team",
        team_options(view, "pitching_code"),
        index=0,
        key="pitcher_mlb_eq_team",
    )
    view = filter_by_team_token(view, "pitching_code", team)
    if view.empty:
        st.info("No rows after filtering.")
        return

    metric_base_cols = [
        col
        for col in PITCHER_COMPS_BASE_FEATURE_COLS + PITCHER_COMPS_EXTRA_FEATURE_COLS
        if col in view.columns and f"{col}_mlb_eq" in view.columns
    ]
    default_metrics = [
        col for col in PITCHER_COMPS_BASE_FEATURE_COLS if col in metric_base_cols
    ]
    selected_metrics = st.multiselect(
        "Metric Columns",
        options=metric_base_cols,
        default=default_metrics,
        key="pitcher_mlb_eq_metrics",
        format_func=lambda col: _pitcher_display_map().get(col, col),
    )
    if not selected_metrics:
        st.info("Select at least one metric column.")
        return

    table_df = view.copy()
    for col in selected_metrics:
        eq_col = f"{col}_mlb_eq"
        delta_col = f"{col}_mlb_delta"
        table_df[delta_col] = table_df[eq_col] - table_df[col]
    table_df = table_df.assign(
        Level=table_df["level_id"].map(lambda v: LEVEL_LABELS.get(int(v), str(int(v)))),
        __season=table_df["season"],
        __level=table_df["level_id"],
    )

    metric_cols: list[str] = []
    for col in selected_metrics:
        metric_cols.extend([col, f"{col}_mlb_eq", f"{col}_mlb_delta"])

    show_cols = [
        "name",
        "pitcher_mlbid",
        "pitching_code",
        "season",
        "Level",
        "TBF",
        "IP",
        "GS",
        *metric_cols,
        "__season",
        "__level",
    ]
    show_cols = [col for col in show_cols if col in table_df.columns]
    table_df = table_df[show_cols].copy()

    rename_map = {
        "name": "Name",
        "pitcher_mlbid": "Player ID",
        "pitching_code": "Team",
        "season": "Season",
    }
    display_map = _pitcher_display_map(include_mlb_eq=True)
    for col in selected_metrics:
        rename_map[col] = display_map.get(col, col)
        rename_map[f"{col}_mlb_eq"] = display_map.get(f"{col}_mlb_eq", f"{col} MLB Eq")
        rename_map[f"{col}_mlb_delta"] = (
            f"{display_map.get(f'{col}_mlb_eq', f'{col} MLB Eq')} Delta"
        )
    table_df = table_df.rename(columns=rename_map)

    reverse_cols = PITCHER_REVERSE_DISPLAY_COLS | {
        f"{name} MLB Eq" for name in PITCHER_REVERSE_DISPLAY_COLS
    }
    render_table(
        table_df,
        reverse_cols=reverse_cols,
        group_cols=["__season", "__level"],
        stats_df=table_df,
        abs_cols=ABS_GRADIENT_COLS_PITCHERS,
    )
    download_button(table_df, "pitcher_mlb_equivalencies", "pitcher_mlb_eq_download")

    if not pitcher_mlb_eq_coeffs.empty:
        with st.expander("Translation coefficients", expanded=False):
            coeff_df = pitcher_mlb_eq_coeffs.copy()
            coeff_df = coeff_df.assign(
                from_level=coeff_df["src_level"].map(
                    lambda v: LEVEL_LABELS.get(int(v), str(int(v)))
                ),
                to_level=coeff_df["dst_level"].map(
                    lambda v: LEVEL_LABELS.get(int(v), str(int(v)))
                ),
            )
            coeff_df["metric"] = coeff_df["metric"].map(
                lambda c: _pitcher_display_map().get(c, c)
            )
            coeff_df = coeff_df.rename(
                columns={
                    "metric": "Metric",
                    "from_level": "From",
                    "to_level": "To",
                    "a": "Intercept (a)",
                    "b": "Rate (b)",
                    "n": "Sample",
                    "fit_type": "Type",
                    "min_src_tbf": "Min TBF (From)",
                    "min_dst_tbf": "Min TBF (To)",
                }
            )
            coeff_cols = [
                "Metric",
                "From",
                "To",
                "Type",
                "Intercept (a)",
                "Rate (b)",
                "Sample",
                "Min TBF (From)",
                "Min TBF (To)",
            ]
            coeff_df = coeff_df[[col for col in coeff_cols if col in coeff_df.columns]]
            render_table(coeff_df, show_controls=False, round_decimals=2)


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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(pitchers_reg_df),
                index=0,
                key="pitcher_ar_game_type_group",
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
                "Filter By", ["IP", "TBF", "GS"], index=1, key="pitcher_ar_filter_type"
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
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitcher_ar_player",
            )
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
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
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            df = pitcher_workload_filter(df, filter_type, min_value)

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "TBF",
                "IP",
                "GS",
                "HR",
                "stuff",
                "grade_v13",
                "fastball_velo_reg",
                "max_velo_reg",
                "fastball_vaa_reg",
                "FA_pct_reg",
                "BB_rpm_reg",
                "SwStr_reg",
                "p_SwStr_pct_reg",
                "p_Swing_pct_reg",
                "Damage_pct_reg",
                "p_Damage_pct_reg",
                "Ball_pct_reg",
                "Z_Contact_reg",
                "Chase_reg",
                "CSW_reg",
                "LA_lte_0_reg",
                "rel_z_reg",
                "rel_x_reg",
                "ext_reg",
                "SBO",
                "SB",
                "takeoff_rate_reg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            if "grade_v13" in df.columns:
                df["grade_v13"] = df["grade_v13"].round(0).astype("Int64")
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "GS": "GS",
                "HR": "HR",
                "stuff": "Pitch Grade",
                "grade_v13": "Execution Grade",
                "fastball_velo_reg": "FA mph",
                "max_velo_reg": "Max FA mph",
                "fastball_vaa_reg": "FA VAA",
                "FA_pct_reg": "FA Usage (%)",
                "BB_rpm_reg": "BB Spin",
                "SwStr_reg": "SwStr (%)",
                "p_SwStr_pct_reg": "pSwStr (%)",
                "p_Swing_pct_reg": "pSwing (%)",
                "Damage_pct_reg": "Damage/BBE (%)",
                "p_Damage_pct_reg": "pDamage/BBE (%)",
                "Ball_pct_reg": "Ball (%)",
                "Z_Contact_reg": "Z-Contact (%)",
                "Chase_reg": "Chase (%)",
                "CSW_reg": "CSW (%)",
                "LA_lte_0_reg": "LA<=0%",
                "rel_z_reg": "Vertical Release (ft.)",
                "rel_x_reg": "Horizontal Release (ft.)",
                "ext_reg": "Extension (ft.)",
                "SBO": "SBO",
                "SB": "SB",
                "takeoff_rate_reg": "Takeoff% Against",
            }
            df = df.rename(columns=rename_map)
            df = maybe_add_level_col(df, level)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)", "Damage/BBE (%)", "pDamage/BBE (%)", "HR", "SBO", "SB", "Takeoff% Against"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCHERS,
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
                game_type_group = st.selectbox(
                    "Game Type",
                    game_type_group_options(pitcher_splits_df),
                    index=0,
                    key=f"pitcher_splits_game_type_group_{idx}",
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
                    ["IP", "TBF", "GS"],
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
                    format_func=lambda v: (
                        "All"
                        if v == "All"
                        else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                    ),
                    key=f"pitcher_splits_player_{idx}",
                )
            with right:
                if game_type_group != "Regular Season":
                    st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
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
                df = filter_by_game_type_group(df, game_type_group)
                df = filter_by_values(df, "split", split_choice)
                df = filter_by_team_token(df, "pitching_code", team)
                df = filter_by_values(df, "pitcher_mlbid", player)
                df = df.assign(__season=df["season"], __level=df["level_id"])

                df = pitcher_workload_filter(df, filter_type, min_value)

                columns = [
                    "name",
                    "pitcher_mlbid",
                    "split",
                    "pitching_code",
                    "season",
                    "TBF",
                    "IP",
                    "GS",
                    "HR",
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
                    "pitcher_mlbid": "Player ID",
                    "pitching_code": "Team",
                    "season": "Season",
                    "split": "Split",
                    "GS": "GS",
                    "HR": "HR",
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
                df = maybe_add_level_col(df, level)
                df = df.sort_values(by="Pitch Grade", ascending=False)
                stats_df = base_stats[
                    [col for col in columns if col in base_stats.columns]
                ].rename(columns=rename_map)
                render_table(
                    df,
                    reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)", "pDamage/BBE (%)", "HR"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                    abs_cols=ABS_GRADIENT_COLS_PITCHERS,
                )
                download_button(
                    df,
                    f"pitcher_splits_{idx}",
                    f"pitcher_splits_download_{idx}",
                )


def pitcher_gamelogs_page():
    """Pitchers - Game Logs page"""
    st.title("Pitcher Game Logs")
    pitcher_gamelogs = get_pitcher_gamelogs()

    if pitcher_gamelogs.empty:
        st.info("Missing pitcher_gamelogs.parquet — run the daily pipeline to generate it.")
        return

    _PITCHER_GL_COLS = [
        "game_date", "pitcher_name", "pitcher_mlbid", "pitching_code",
        "pitcher_hand", "game_pk", "opp_team",
        "TBF", "bbe", "pitches", "whiffs", "chases", "FA_mph", "stuff", "grade_v13",
        "HR", "XBH", "hits", "damaged_bbe",
        "la_gte_20_bbe", "la_lte_0_bbe", "BB", "K",
        "strikes", "balls", "swings",
        "zone_pitches", "out_of_zone", "FA", "BR", "OFF",
        "vs_LHB", "vs_RHB",
    ]
    _RENAME = {
        "game_date": "Date", "pitcher_name": "Name",
        "pitcher_mlbid": "Player ID", "pitching_code": "Team",
        "pitcher_hand": "Hand", "game_pk": "Game ID",
        "opp_team": "vs",
        "bbe": "BBE", "damaged_bbe": "Damage BBE", "hits": "H",
        "la_gte_20_bbe": "LA >= 20", "la_lte_0_bbe": "LA<=0",
        "swings": "Swings", "chases": "Chases", "whiffs": "Whiffs",
        "pitches": "Pitches",
        "zone_pitches": "Zone", "out_of_zone": "Out of Zone",
        "strikes": "Strikes", "balls": "Balls",
        "FA": "FA#", "BR": "BR#", "OFF": "OFF#",
        "FA_mph": "Avg FA mph",
        "vs_LHB": "vs LHB", "vs_RHB": "vs RHB",
        "stuff": "Pitch Grade", "grade_v13": "Exec Grade",
    }
    _level_map = {
        "All": [1, 11, 14, 16], "MLB": [1], "Triple-A": [11],
        "Low-A": [14], "Low Minors": [16],
    }

    tab_date, tab_player = st.tabs(["By Date", "By Player"])

    with tab_date:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Level", list(_level_map.keys()), index=1, key="pgl_date_level"
            )
            season = st.multiselect(
                "Season", season_options(pitcher_gamelogs),
                default=(
                    [season_options(pitcher_gamelogs)[1]]
                    if len(season_options(pitcher_gamelogs)) > 1 else ["All"]
                ),
                key="pgl_date_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(pitcher_gamelogs),
                index=0, key="pgl_date_gtg",
            )
            base = pitcher_gamelogs[
                pitcher_gamelogs["level_id"].isin(_level_map[level])
            ]
            base = filter_by_values(base, "season", season)
            base = filter_by_game_type_group(base, game_type_group)
            dates = sorted(base["game_date"].dropna().astype(str).unique(), reverse=True)
            date_choice = st.selectbox(
                "Date", ["All"] + (dates if dates else ["(none)"]), index=0, key="pgl_date_date",
                format_func=lambda d: (
                    pd.to_datetime(d).strftime("%m/%d/%Y") if d not in ("All", "(none)", "") else d
                ),
            )
            team = st.selectbox(
                "Team", team_options(base, "pitching_code"), index=0, key="pgl_date_team"
            )
        with right:
            df = base.copy() if date_choice == "All" else base[base["game_date"].astype(str) == date_choice].copy()
            df = filter_by_team_token(df, "pitching_code", team)
            df = df[[c for c in _PITCHER_GL_COLS if c in df.columns]].copy()
            if "game_date" in df.columns:
                _sort = ["game_date", "pitcher_name"] if "pitcher_name" in df.columns else ["game_date"]
                df = df.sort_values(_sort, ascending=[False] + [True] * (len(_sort) - 1))
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
            df = df.rename(columns=_RENAME)
            render_table(df, stats_df=pd.DataFrame())
            download_button(df, "pitcher_gamelogs_date", "pgl_date_dl")

    with tab_player:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Level", list(_level_map.keys()), index=1, key="pgl_pl_level"
            )
            season = st.multiselect(
                "Season", season_options(pitcher_gamelogs),
                default=(
                    [season_options(pitcher_gamelogs)[1]]
                    if len(season_options(pitcher_gamelogs)) > 1 else ["All"]
                ),
                key="pgl_pl_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(pitcher_gamelogs),
                index=0, key="pgl_pl_gtg",
            )
            base = pitcher_gamelogs[
                pitcher_gamelogs["level_id"].isin(_level_map[level])
            ]
            base = filter_by_values(base, "season", season)
            base = filter_by_game_type_group(base, game_type_group)
            player_opts, player_name_map = player_id_options(
                base, "pitcher_mlbid", "pitcher_name"
            )
            player_vals = [v for v in player_opts if v != "All"]
            player_choice = st.selectbox(
                "Player", player_vals if player_vals else ["(none)"],
                index=0,
                format_func=lambda v: f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                if v != "(none)" else "(none)",
                key="pgl_pl_player",
            )
        with right:
            if not player_vals:
                st.info("No players available.")
            else:
                df = base[base["pitcher_mlbid"] == player_choice].copy()
                player_cols = [c for c in _PITCHER_GL_COLS if c not in ("pitcher_name", "pitcher_mlbid")]
                df = df[[c for c in player_cols if c in df.columns]].copy()
                df = df.sort_values("game_date", ascending=False) if "game_date" in df.columns else df
                if "game_date" in df.columns:
                    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
                df = df.rename(columns=_RENAME)
                render_table(df, stats_df=pd.DataFrame())
                download_button(df, "pitcher_gamelogs_player", "pgl_pl_dl")
