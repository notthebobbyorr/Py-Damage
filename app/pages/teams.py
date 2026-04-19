from __future__ import annotations

import streamlit as st

from app.config import (
    GAME_TYPE_GROUP_NOTE,
    HIGHER_IS_WORSE_COLS,
)
from app.datasets import (
    team_damage,
    team_stuff,
)
from app.filters import (
    download_button,
    filter_by_game_type_group,
    filter_by_values,
    game_type_group_options,
    season_options,
)
from app.viz import render_table


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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(team_damage),
                index=0,
                key="team_hitting_game_type_group",
            )
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
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
            df = filter_by_game_type_group(df, game_type_group)
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
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(team_stuff),
                index=0,
                key="team_pitching_game_type_group",
            )
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
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
            df = filter_by_game_type_group(df, game_type_group)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "pitching_code",
                "season",
                "IP",
                "stuff",
                "grade_v13",
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
            if "grade_v13" in df.columns:
                df["grade_v13"] = df["grade_v13"].round(0).astype("Int64")
            rename_map = {
                "pitching_code": "Team",
                "season": "Season",
                "stuff": "Pitch Grade",
                "grade_v13": "Execution Grade",
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
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)", "pDamage/BBE (%)", "pDamage+Loc/BBE (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                include_team_label=True,
            )
            download_button(df, "team_pitching", "team_pitching_download")
