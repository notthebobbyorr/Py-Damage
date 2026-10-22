from __future__ import annotations
from app.filters import split_workload_controls, numeric_filter, apply_workload_maximum

import streamlit as st

from app.config import (
    ABS_GRADIENT_COLS_PITCH_TYPES,
    GAME_TYPE_GROUP_NOTE,
)
from app.datasets import (
    hitting_avg,
    league_pitch_types,
    pitching_avg,
)
from app.filters import (
    download_button,
    filter_by_game_type_group,
    filter_by_values,
    game_type_group_options,
    season_options,
)
from app.viz import render_table
from app.utils import maybe_add_level_col
from app.hitter_pitch_splits import BAT_TRACKING_COLUMNS


def league_hitting():
    """League - Hitting Stats page"""
    st.title("League Averages - Hitting")

    tab_season, tab_pitch_types = st.tabs(["Season Stats", "Pitch Type Splits"])
    with tab_season:
        _league_hitting_season()
    with tab_pitch_types:
        from app.hitting_pitch_split_tables import render_hitting_pitch_splits

        render_hitting_pitch_splits()


def _league_hitting_season():

    if hitting_avg.empty:
        st.info("Missing new_hitting_lg_avg.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level", ["MLB", "Triple-A", "Low-A", "Low Minors", "All"],
                index=0, key="lg_hit_level",
            )
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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(hitting_avg),
                index=0,
                key="league_hitting_game_type_group",
            )
            workload_column, minimum = split_workload_controls(hitting_avg,'_league_hitting_season_workload',pitching=False,default_minimum=0)
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
            df = hitting_avg.copy()
            df = numeric_filter(df,workload_column,minimum)
            df = apply_workload_maximum(df,'_league_hitting_season_workload')
            df = filter_by_values(df, "level_id", [1, 11, 14, 16] if level == "All" else {"MLB": 1, "Triple-A": 11, "Low-A": 14, "Low Minors": 16}[level])
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
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
                'whiff_pct', 'Swing_pct', 'SwStr',

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
                *BAT_TRACKING_COLUMNS,
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                'whiff_pct': 'Whiff (%)', 'Swing_pct': 'Swing (%)', 'SwStr': 'SwStr (%)',
                "miss_distance": "Avg Miss Distance (in.)",
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
                **BAT_TRACKING_COLUMNS,
            }
            df = df.rename(columns=rename_map)
            if "Level" not in df.columns:
                df = maybe_add_level_col(df, level)
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            df = render_table(
                df, show_maximum_filters=False,
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                conditional_formatting=False,
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
                ["MLB", "Triple-A", "Low-A", "Low Minors", "All"],
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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(pitching_avg),
                index=0,
                key="league_pitching_game_type_group",
            )
            workload_column, minimum = split_workload_controls(pitching_avg,'league_pitching_workload',pitching=True,default_minimum=0)
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
                "All": [1, 11, 14, 16],
            }
            base_stats = pitching_avg.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitching_avg.copy()
            df = numeric_filter(df,workload_column,minimum)
            df = apply_workload_maximum(df,'league_pitching_workload')
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                'whiff_pct', 'Swing_pct',
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
                "miss_distance", "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round BB_rpm and stuff to integers
            if "BB_rpm" in df.columns:
                df["BB_rpm"] = df["BB_rpm"].round(0)
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            rename_map = {
                'whiff_pct': 'Whiff (%)', 'Swing_pct': 'Swing (%)',
                "miss_distance": "Avg Miss Distance (in.)",
                "season": "Season",
                "CSW": "CSW (%)",
                "Ball_pct": "Ball (%)",
                "SwStr": "SwStr (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "fastball_velo": "FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "BB_rpm": "BR spin",
                "stuff": "Pitch Grade",
                "stuff_z": "Pitch Grade Z",
                "LA_lte_0": "LA<=0%",
            }
            df = df.rename(columns=rename_map)
            if "Level" not in df.columns:
                df = maybe_add_level_col(df, level)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            df = render_table(
                df, show_maximum_filters=False,
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                conditional_formatting=False,
             workload_kind="pitcher")
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
        level = st.selectbox(
            "Select Level", ["MLB", "Triple-A", "Low-A", "Low Minors", "All"],
            index=0, key="lg_pitch_types_level",
        )
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
        game_type_group = st.selectbox(
            "Game Type",
            game_type_group_options(league_pitch_types),
            index=0,
            key="league_pitch_types_game_type_group",
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
        workload_column, minimum = split_workload_controls(league_pitch_types,'league_pitch_level_workload',pitching=True,default_minimum=0)
    with right:
        if game_type_group != "Regular Season":
            st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
        base_stats = league_pitch_types.copy()
        base_stats = base_stats.assign(__season=base_stats["season"])
        df = league_pitch_types.copy()
        df = numeric_filter(df,workload_column,minimum)
        df = apply_workload_maximum(df,'league_pitch_level_workload')
        df = filter_by_values(df, "level_id", [1, 11, 14, 16] if level == "All" else {"MLB": 1, "Triple-A": 11, "Low-A": 14, "Low Minors": 16}[level])
        df = filter_by_values(df, "season", season)
        df = filter_by_game_type_group(df, game_type_group)
        df = filter_by_values(df, "throws", throws)
        df = filter_by_values(df, "pitch_tag", pitch_tag)
        df = df.assign(__season=df["season"], __level=df["level_id"])

        columns = [
            'whiff_pct', 'Swing_pct',
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
            "miss_distance", "__season",
            "__level",
        ]
        df = df[[col for col in columns if col in df.columns]].copy()
        rename_map = {
            'whiff_pct': 'Whiff (%)', 'Swing_pct': 'Swing (%)',
            "miss_distance": "Avg Miss Distance (in.)",
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
        df = maybe_add_level_col(df, level)
        stats_df = base_stats[
            [col for col in columns if col in base_stats.columns]
        ].rename(columns=rename_map)
        df = render_table(
            df, show_maximum_filters=False,
            reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA", "pDamage/BBE (%)", "pDamage+Loc/BBE (%)"},
            group_cols=["__season"],
            stats_df=stats_df,
            conditional_formatting=False,
         workload_kind="pitcher")
        download_button(df, "league_pitch_types", "league_pitch_types_download")
