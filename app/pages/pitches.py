from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import (
    ABS_GRADIENT_COLS_PITCH_TYPES,
    GAME_TYPE_GROUP_NOTE,
)
from app.datasets import (
    pitch_type_gamelogs,
    pitch_type_splits_df,
    pitch_types,
    pitch_types_pct,
    pitch_types_reg_df,
)
from app.filters import (
    download_button,
    filter_by_game_type_group,
    filter_by_team_token,
    filter_by_values,
    game_type_group_options,
    player_id_options,
    season_options,
    team_options,
)
from app.utils import maybe_add_level_col
from app.viz import render_table


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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(pitch_types),
                index=0,
                key="pitch_types_shapes_game_type_group",
            )
            min_pitches = st.number_input(
                "Minimum # Pitches",
                min_value=0,
                max_value=1000,
                value=5,
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
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
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
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
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
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_group", pitch_group)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "pitch_tag",
                "pitches",
                "pct",
                "stuff",
                "grade_v13",
                "velo",
                "max_velo",
                "vaa",
                "haa",
                "vbreak",
                "hbreak",
                "SwStr",
                "p_SwStr_pct",
                "Swing_pct",
                "p_Swing_pct",
                "Damage_pct",
                "p_Damage_pct",
                "HR",
                "LA_lte_0",
                "Z_Contact",
                "Ball_pct",
                "Zone",
                "Chase",
                "CSW",
                "rel_z",
                "rel_x",
                "ext",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round stuff and grade_v13 to integer
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            if "grade_v13" in df.columns:
                df["grade_v13"] = df["grade_v13"].round(0).astype("Int64")
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "pitch_tag": "Pitch Type",
                "pitches": "#",
                "pct": "Usage (%)",
                "stuff": "Pitch Grade",
                "grade_v13": "Execution Grade",
                "velo": "Velo",
                "max_velo": "Max Velo",
                "vaa": "VAA",
                "haa": "HAA",
                "vbreak": "IVB (in.)",
                "hbreak": "HB (in.)",
                "CSW": "CSW (%)",
                "SwStr": "SwStr (%)",
                "p_SwStr_pct": "pSwStr (%)",
                "Swing_pct": "Swing (%)",
                "p_Swing_pct": "pSwing (%)",
                "Damage_pct": "Damage/BBE (%)",
                "p_Damage_pct": "pDamage/BBE (%)",
                "HR": "HR",
                "LA_lte_0": "LA<=0%",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "Zone": "Zone (%)",
                "Ball_pct": "Ball (%)",
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
                reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA", "Damage/BBE (%)", "pDamage/BBE (%)", "HR"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(pitch_types_reg_df),
                index=0,
                key="pitch_types_ar_game_type_group",
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
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
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
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
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
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_group", pitch_group)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "pitch_tag",
                "pitches",
                "pct",
                "stuff",
                "grade_v13",
                "velo_reg",
                "max_velo_reg",
                "vaa_reg",
                "haa_reg",
                "vbreak_reg",
                "hbreak_reg",
                "SwStr_reg",
                "p_SwStr_pct_reg",
                "p_Swing_pct_reg",
                "Damage_pct_reg",
                "HR",
                "LA_lte_0_reg",
                "Z_Contact_reg",
                "Ball_pct_reg",
                "Chase_reg",
                "CSW_reg",
                "rel_z_reg",
                "rel_x_reg",
                "ext_reg",
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
                "pitch_tag": "Pitch Type",
                "pitches": "#",
                "pct": "Usage (%)",
                "stuff": "Pitch Grade",
                "grade_v13": "Execution Grade",
                "velo_reg": "Velo",
                "max_velo_reg": "Max Velo",
                "vaa_reg": "VAA",
                "haa_reg": "HAA",
                "vbreak_reg": "IVB (in.)",
                "hbreak_reg": "HB (in.)",
                "CSW_reg": "CSW (%)",
                "SwStr_reg": "SwStr (%)",
                "p_SwStr_pct_reg": "pSwStr (%)",
                "p_Swing_pct_reg": "pSwing (%)",
                "Damage_pct_reg": "Damage/BBE (%)",
                "HR": "HR",
                "LA_lte_0_reg": "LA<=0%",
                "Z_Contact_reg": "Z-Contact (%)",
                "Chase_reg": "Chase (%)",
                "Ball_pct_reg": "Ball (%)",
                "rel_z_reg": "Vertical Release (ft.)",
                "rel_x_reg": "Horizontal Release (ft.)",
                "ext_reg": "Extension (ft.)",
            }
            df = df.rename(columns=rename_map)
            df = maybe_add_level_col(df, level)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA", "Damage/BBE (%)", "HR"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(pitch_types_pct),
                index=0,
                key="pitch_types_pct_game_type_group",
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
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
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
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
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
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]

            columns = [
                "name",
                "pitcher_mlbid",
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
                "HR",
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
                "HR": "HR",
            }
            df = df.rename(columns=rename_map)
            df = maybe_add_level_col(df, level)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            render_table(
                df,
                reverse_cols={"VAA", "Ball (%)", "Z-Contact (%)", "HR"},
                abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
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
                game_type_group = st.selectbox(
                    "Game Type",
                    game_type_group_options(pitch_type_splits_df),
                    index=0,
                    key=f"pitch_types_splits_game_type_group_{idx}",
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
                    format_func=lambda v: (
                        "All"
                        if v == "All"
                        else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                    ),
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
                df = filter_by_values(df, "pitcher_mlbid", pitcher)
                if "pitch_group" in df.columns:
                    df = filter_by_values(df, "pitch_group", pitch_group)
                df = filter_by_values(df, "pitch_tag", pitch_tag)
                df = df[df["pitches"] >= min_pitches]
                df = df.assign(__season=df["season"], __level=df["level_id"])

                columns = [
                    "name",
                    "pitcher_mlbid",
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
                    "Damage_pct",
                    "HR",
                    "Z_Contact",
                    "Ball_pct",
                    "Zone",
                    "Chase",
                    "CSW",
                    "rel_z",
                    "rel_x",
                    "ext",
                    "__season",
                    "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                if "stuff" in df.columns:
                    df["stuff"] = df["stuff"].round(0)
                rename_map = {
                    "name": "Name",
                    "pitcher_mlbid": "Player ID",
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
                    "Damage_pct": "Damage/BBE (%)",
                    "HR": "HR",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "Zone": "Zone (%)",
                    "Ball_pct": "Ball (%)",
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
                    reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA", "Damage/BBE (%)", "HR"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                    abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
                    label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
                )
                download_button(
                    df,
                    f"pitch_splits_{idx}",
                    f"pitch_splits_download_{idx}",
                )


def pitch_type_gamelogs_page():
    """Individual Pitches - Game Logs page"""
    st.title("Pitch Type Game Logs")

    if pitch_type_gamelogs.empty:
        st.info("Missing pitch_type_gamelogs.parquet — run the daily pipeline to generate it.")
        return

    _PT_GL_COLS = [
        "game_date", "pitcher_name", "pitcher_mlbid", "pitching_code",
        "pitch_tag", "pitcher_hand", "game_pk", "opp_team",
        "bbe", "HR", "XBH", "hits", "damaged_bbe",
        "la_gte_20_bbe", "la_lte_0_bbe", "BB", "K",
        "pitches", "strikes", "balls", "swings", "whiffs", "chases",
        "zone_pitches", "out_of_zone",
        "velo", "vs_LHB", "vs_RHB",
    ]
    _RENAME = {
        "game_date": "Date", "pitcher_name": "Name",
        "pitcher_mlbid": "Player ID", "pitching_code": "Team",
        "pitch_tag": "Pitch Type", "pitcher_hand": "Hand", "game_pk": "Game ID",
        "opp_team": "vs",
        "bbe": "BBE", "damaged_bbe": "Damage BBE", "hits": "H",
        "la_gte_20_bbe": "LA >= 20", "la_lte_0_bbe": "LA<=0",
        "swings": "Swings", "chases": "Chases", "whiffs": "Whiffs",
        "pitches": "Pitches",
        "zone_pitches": "Zone", "out_of_zone": "Out of Zone",
        "strikes": "Strikes", "balls": "Balls",
        "velo": "Avg mph",
        "vs_LHB": "vs LHB", "vs_RHB": "vs RHB",
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
                "Level", list(_level_map.keys()), index=1, key="ptgl_date_level"
            )
            season = st.multiselect(
                "Season", season_options(pitch_type_gamelogs),
                default=(
                    [season_options(pitch_type_gamelogs)[1]]
                    if len(season_options(pitch_type_gamelogs)) > 1 else ["All"]
                ),
                key="ptgl_date_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(pitch_type_gamelogs),
                index=0, key="ptgl_date_gtg",
            )
            base = pitch_type_gamelogs[
                pitch_type_gamelogs["level_id"].isin(_level_map[level])
            ]
            base = filter_by_values(base, "season", season)
            base = filter_by_game_type_group(base, game_type_group)
            dates = sorted(base["game_date"].dropna().astype(str).unique(), reverse=True)
            date_choice = st.selectbox(
                "Date", ["All"] + (dates if dates else ["(none)"]), index=0, key="ptgl_date_date",
                format_func=lambda d: (
                    pd.to_datetime(d).strftime("%m/%d/%Y") if d not in ("All", "(none)", "") else d
                ),
            )
            team = st.selectbox(
                "Team", team_options(base, "pitching_code"), index=0, key="ptgl_date_team"
            )
        with right:
            df = base.copy() if date_choice == "All" else base[base["game_date"].astype(str) == date_choice].copy()
            df = filter_by_team_token(df, "pitching_code", team)
            df = df[[c for c in _PT_GL_COLS if c in df.columns]].copy()
            if "game_date" in df.columns:
                _sort = ["game_date", "pitcher_name", "pitch_tag"] if "pitcher_name" in df.columns else ["game_date"]
                df = df.sort_values(_sort, ascending=[False] + [True] * (len(_sort) - 1))
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
            df = df.rename(columns=_RENAME)
            render_table(df, stats_df=pd.DataFrame())
            download_button(df, "pitch_type_gamelogs_date", "ptgl_date_dl")

    with tab_player:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Level", list(_level_map.keys()), index=1, key="ptgl_pl_level"
            )
            season = st.multiselect(
                "Season", season_options(pitch_type_gamelogs),
                default=(
                    [season_options(pitch_type_gamelogs)[1]]
                    if len(season_options(pitch_type_gamelogs)) > 1 else ["All"]
                ),
                key="ptgl_pl_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(pitch_type_gamelogs),
                index=0, key="ptgl_pl_gtg",
            )
            base = pitch_type_gamelogs[
                pitch_type_gamelogs["level_id"].isin(_level_map[level])
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
                key="ptgl_pl_player",
            )
        with right:
            if not player_vals:
                st.info("No players available.")
            else:
                df = base[base["pitcher_mlbid"] == player_choice].copy()
                player_cols = [c for c in _PT_GL_COLS if c not in ("pitcher_name", "pitcher_mlbid")]
                df = df[[c for c in player_cols if c in df.columns]].copy()
                df = df.sort_values(
                    ["game_date", "pitch_tag"], ascending=[False, True]
                ) if "game_date" in df.columns else df
                if "game_date" in df.columns:
                    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
                df = df.rename(columns=_RENAME)
                render_table(df, stats_df=pd.DataFrame())
                download_button(df, "pitch_type_gamelogs_player", "ptgl_pl_dl")
