from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.config import (
    ABS_GRADIENT_COLS_PITCH_TYPES,
    GAME_TYPE_GROUP_NOTE,
    PITCH_COMPS_BASE_FEATURE_COLS,
    PITCH_COMPS_EXTRA_FEATURE_COLS,
    PITCH_REVERSE_DISPLAY_COLS,
)
from app.datasets import (
    get_pitch_type_gamelogs,
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
from app.utils import (
    _pitch_display_map,
    _similarity_choice_labels,
    maybe_add_level_col,
    rank_for_display,
)
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
                "z_angle_release",
                "x_angle_release",
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
                "z_angle_release": "VRA",
                "x_angle_release": "HRA",
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
        _LEVEL_MAP = {
            "All": [1, 11, 14, 16],
            "MLB": [1],
            "Triple-A": [11],
            "Low-A": [14],
            "Low Minors": [16],
        }
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
                max_value=2000,
                value=20,
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
            df = pitch_types_pct.copy()
            df = df[df["level_id"].isin(_LEVEL_MAP[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]

            df = rank_for_display(df, [
                "pct", "stuff", "velo", "max_velo", "vaa", "haa",
                "vbreak", "hbreak", "SwStr", "LA_lte_0", "Ball_pct",
                "Z_Contact", "Chase", "CSW",
                "p_SwStr_pct", "Damage_pct", "p_Damage_pct", "takeoff_rate",
            ], ["season", "level_id", "game_type_group", "pitch_tag"],
            reverse_cols={"vaa", "Ball_pct", "Z_Contact", "takeoff_rate", "Damage_pct", "p_Damage_pct"})

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
                "grade_v13",
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
                "grade_v13": "Execution Grade",
                "p_SwStr_pct_pctile": "pSwStr (%)",
                "Damage_pct_pctile": "Damage/BBE%",
                "p_Damage_pct_pctile": "pDamage/BBE%",
                "takeoff_rate_pctile": "Takeoff Against (%)",
            }
            df = df.rename(columns=rename_map)
            df = maybe_add_level_col(df, level)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            render_table(
                df,
                reverse_cols={"HR"},
                abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
                label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
                round_decimals=0,
            )
            download_button(df, "pitch_percentiles", "pitch_pct_download")


def pitch_comps():
    """Individual Pitches - Pitch Level Comps page"""
    st.title("Pitch Level Comparisons")

    if pitch_types.empty:
        st.info("Missing new_pitch_types.csv")
        return

    comp_df = pitch_types.copy()
    if "game_type_group" in comp_df.columns:
        comp_df = comp_df[comp_df["game_type_group"] != "Spring Training"]

    # Fall back to model-estimated arm_angle_right when Statcast didn't publish it.
    if "arm_angle_right" in comp_df.columns and "inf_arm_angle" in comp_df.columns:
        comp_df = comp_df.copy()
        comp_df["arm_angle_right_imputed"] = (
            comp_df["arm_angle_right"].isna() & comp_df["inf_arm_angle"].notna()
        )
        comp_df["arm_angle_right"] = comp_df["arm_angle_right"].fillna(
            comp_df["inf_arm_angle"]
        )

    target_pool = comp_df[
        (comp_df["level_id"] == 1) & (comp_df["pitches"] >= 5)
    ].copy()
    eligible_all = comp_df[
        (comp_df["level_id"] == 1) & (comp_df["pitches"] >= 100)
    ].copy()

    if target_pool.empty:
        st.info("No eligible MLB pitch-seasons (min 5 pitches).")
        return
    if eligible_all.empty:
        st.info("No eligible MLB comparison pitch-seasons (min 100 pitches).")
        return

    seasons = season_options(target_pool, "season")[1:]
    if not seasons:
        st.info("No seasons available.")
        return
    season_choice = st.selectbox("Season", seasons, index=0, key="pitch_comps_season")
    season_df = target_pool[target_pool["season"] == season_choice]
    if season_df.empty:
        st.info("No pitch rows for this season selection.")
        return

    player_options, player_name_map = player_id_options(
        season_df, "pitcher_mlbid", "name"
    )
    player_values = [opt for opt in player_options if opt != "All"]
    if not player_values:
        st.info("No players available for this filter.")
        return
    player_choice = st.selectbox(
        "Pitcher",
        player_values,
        index=0,
        format_func=lambda v: f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
        key="pitch_comps_player",
    )
    player_df = season_df[season_df["pitcher_mlbid"] == player_choice]
    if player_df.empty:
        st.info("No pitches found for that pitcher in this season.")
        return

    pitch_tags = sorted(player_df["pitch_tag"].dropna().unique().tolist())
    if not pitch_tags:
        st.info("No pitch types available for this pitcher.")
        return
    pitch_tag_choice = st.selectbox(
        "Pitch Type",
        pitch_tags,
        index=0,
        key="pitch_comps_pitch_tag",
    )
    target_df = player_df[player_df["pitch_tag"] == pitch_tag_choice]
    if target_df.empty:
        st.info("No target row found for that selection.")
        return

    display_map = _pitch_display_map()
    allowed_cols = list(
        dict.fromkeys(PITCH_COMPS_BASE_FEATURE_COLS + PITCH_COMPS_EXTRA_FEATURE_COLS)
    )
    exclude_cols = {
        "pitcher_mlbid",
        "level_id",
        "game_pk",
        "pitches",
        "pitches_n",
        "pitches_num",
        "pitches_den",
        "season",
        "pct",
    }
    numeric_cols, similarity_labels = _similarity_choice_labels(
        eligible_all, display_map, exclude_cols
    )
    numeric_cols = [col for col in numeric_cols if col in allowed_cols]
    default_feature_cols = [
        col for col in PITCH_COMPS_BASE_FEATURE_COLS if col in numeric_cols
    ]

    feature_cols = st.multiselect(
        "Similarity Score Columns",
        options=numeric_cols,
        default=default_feature_cols,
        key="pitch_comps_similarity_cols",
        format_func=lambda col: similarity_labels.get(col, col),
    )
    feature_cols = [col for col in feature_cols if col in numeric_cols]
    feature_cols = list(dict.fromkeys(feature_cols))
    if not feature_cols:
        st.info("Select at least one column to compute similarity scores.")
        return

    # Exclude the target row from the comparison pool
    eligible_comp = eligible_all.copy()
    eligible_comp = eligible_comp[
        ~(
            (eligible_comp["pitcher_mlbid"] == player_choice)
            & (eligible_comp["season"] == season_choice)
            & (eligible_comp["pitch_tag"] == pitch_tag_choice)
        )
    ]
    eligible_comp = eligible_comp[eligible_comp[feature_cols].notna().any(axis=1)]
    if eligible_comp.empty:
        st.info("No comparable pitches found.")
        return

    stats = eligible_comp[feature_cols].copy()
    means = stats.mean().fillna(0.0)
    stats = stats.fillna(means)
    stds = stats.std(ddof=0).replace(0, np.nan)
    zscores = ((stats - means) / stds).fillna(0)
    target_stats = target_df[feature_cols].copy().fillna(means)
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

    display_cols = [
        "name",
        "pitching_code",
        "season",
        "pitch_tag",
        "pitches",
        "similarity_score",
        *feature_cols,
        "arm_angle_right_imputed",
        "__season",
        "__level",
    ]
    df = eligible_comp[
        [col for col in display_cols if col in eligible_comp.columns]
    ].copy()
    if "grade_v13" in df.columns:
        df["grade_v13"] = df["grade_v13"].round(0).astype("Int64")
    if "arm_angle_right_imputed" in df.columns:
        df["arm_angle_right_imputed"] = df["arm_angle_right_imputed"].map(
            lambda v: "*" if bool(v) else ""
        )
    df = df.rename(columns={**display_map, **similarity_labels, "arm_angle_right_imputed": "Inferred"})
    df = df.loc[:, ~df.columns.duplicated()]

    stats_df = eligible_all.copy()
    stats_df = stats_df.assign(
        __season=stats_df["season"], __level=stats_df["level_id"]
    )
    stats_columns = [
        "name",
        "pitching_code",
        "season",
        "pitch_tag",
        "pitches",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season",
        "__level",
    ]
    stats_df = stats_df[
        [col for col in stats_columns if col in stats_df.columns]
    ].copy()
    if "grade_v13" in stats_df.columns:
        stats_df["grade_v13"] = stats_df["grade_v13"].round(0).astype("Int64")
    stats_df = stats_df.rename(columns={**display_map, **similarity_labels})
    stats_df = stats_df.loc[:, ~stats_df.columns.duplicated()]

    target_cols = [
        "name",
        "pitching_code",
        "season",
        "pitch_tag",
        "pitches",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "arm_angle_right_imputed",
        "__season",
        "__level",
    ]
    target_view = target_df.assign(
        __season=target_df["season"], __level=target_df["level_id"]
    )
    target_view = target_view[
        [col for col in target_cols if col in target_view.columns]
    ].copy()
    if "grade_v13" in target_view.columns:
        target_view["grade_v13"] = target_view["grade_v13"].round(0).astype("Int64")
    if "arm_angle_right_imputed" in target_view.columns:
        target_view["arm_angle_right_imputed"] = target_view["arm_angle_right_imputed"].map(
            lambda v: "*" if bool(v) else ""
        )
    target_view = target_view.rename(columns={**display_map, **similarity_labels, "arm_angle_right_imputed": "Inferred"})
    target_view = target_view.loc[:, ~target_view.columns.duplicated()]

    st.caption("Selected pitch")
    if "Inferred" in df.columns and (df["Inferred"] == "*").any():
        st.caption(
            "Rows with `*` in the **Inferred** column use a model-estimated arm angle "
            "(applied when Statcast hasn't published the real value for that season)."
        )
    render_table(
        target_view,
        reverse_cols=PITCH_REVERSE_DISPLAY_COLS,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
        show_controls=False,
        label_cols=["Name", "Pitch Type"],
    )
    st.caption("Most similar pitches (MLB, min 100 pitches)")
    render_table(
        df,
        reverse_cols=PITCH_REVERSE_DISPLAY_COLS,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
        label_cols=["Name", "Pitch Type"],
        default_sort_col="Similarity (0-100)",
    )
    download_button(df, "pitch_comps", "pitch_comps_download")


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
    pitch_type_gamelogs = get_pitch_type_gamelogs()

    if pitch_type_gamelogs.empty:
        st.info("Missing pitch_type_gamelogs.parquet — run the daily pipeline to generate it.")
        return

    _PT_GL_COLS = [
        "game_date", "pitcher_name", "pitcher_mlbid", "pitching_code",
        "pitch_tag", "pitcher_hand", "game_pk", "opp_team",
        "bbe", "pitches", "whiffs", "chases", "velo", "stuff", "grade_v13",
        "HR", "XBH", "hits", "damaged_bbe",
        "la_gte_20_bbe", "la_lte_0_bbe", "BB", "K",
        "strikes", "balls", "swings",
        "zone_pitches", "out_of_zone",
        "vs_LHB", "vs_RHB",
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
