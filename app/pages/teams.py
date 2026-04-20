from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import (
    GAME_TYPE_GROUP_NOTE,
    HIGHER_IS_WORSE_COLS,
)
from app.datasets import (
    team_damage,
    team_hitter_gamelogs,
    team_pitcher_gamelogs,
    team_stuff,
)
from app.filters import (
    download_button,
    filter_by_game_type_group,
    filter_by_team_token,
    filter_by_values,
    game_type_group_options,
    season_options,
    team_options,
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


def team_hitting_gamelogs():
    """Teams - Hitting Game Logs page"""
    st.title("Team Hitting Game Logs")

    if team_hitter_gamelogs.empty:
        st.info("Missing hitter_gamelogs.parquet — run the daily pipeline to generate it.")
        return

    _T_H_GL_COLS = [
        "game_date", "hitting_code", "game_pk", "opp_team",
        "PA", "bbe", "HR", "XBH", "hits", "damaged_bbe",
        "pulled_fbs", "la_gte_20_bbe", "la_lte_0_bbe", "BB", "K",
        "pitches", "FA", "BR", "OFF", "swings", "chases", "whiffs",
        "selective_takes", "hittable_takes", "vs_RHP", "vs_LHP",
    ]
    _RENAME = {
        "game_date": "Date", "hitting_code": "Team",
        "game_pk": "Game ID", "opp_team": "vs",
        "bbe": "BBE", "damaged_bbe": "Damage BBE", "hits": "H",
        "pulled_fbs": "Pulled FBs",
        "la_gte_20_bbe": "LA>=20", "la_lte_0_bbe": "LA<=0",
        "pitches": "Pitches", "FA": "FA#", "BR": "BR#", "OFF": "OFF#",
        "swings": "Swings", "chases": "Chases", "whiffs": "Whiffs",
        "selective_takes": "Selective Takes", "hittable_takes": "Hittable Takes",
        "vs_RHP": "vs RHP", "vs_LHP": "vs LHP",
    }
    _level_map = {
        "MLB": [1], "Triple-A": [11], "Low-A": [14], "Low Minors": [16],
    }

    tab_date, tab_team = st.tabs(["By Date", "By Team"])

    with tab_date:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Level", list(_level_map.keys()), index=0, key="thgl_date_level"
            )
            season = st.multiselect(
                "Season", season_options(team_hitter_gamelogs),
                default=(
                    [season_options(team_hitter_gamelogs)[1]]
                    if len(season_options(team_hitter_gamelogs)) > 1 else ["All"]
                ),
                key="thgl_date_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(team_hitter_gamelogs),
                index=0, key="thgl_date_gtg",
            )
            base = team_hitter_gamelogs[
                team_hitter_gamelogs["level_id"].isin(_level_map[level])
            ]
            base = filter_by_values(base, "season", season)
            base = filter_by_game_type_group(base, game_type_group)
            dates = sorted(base["game_date"].dropna().astype(str).unique(), reverse=True)
            date_choice = st.selectbox(
                "Date", ["All"] + (dates if dates else ["(none)"]), index=0, key="thgl_date_date",
                format_func=lambda d: (
                    pd.to_datetime(d).strftime("%m/%d/%Y") if d not in ("All", "(none)", "") else d
                ),
            )
            team = st.selectbox(
                "Team", team_options(base, "hitting_code"), index=0, key="thgl_date_team"
            )
        with right:
            df = base.copy() if date_choice == "All" else base[base["game_date"].astype(str) == date_choice].copy()
            df = filter_by_team_token(df, "hitting_code", team)
            df = df[[c for c in _T_H_GL_COLS if c in df.columns]].copy()
            if "game_date" in df.columns:
                _sort = ["game_date", "hitting_code"] if "hitting_code" in df.columns else ["game_date"]
                df = df.sort_values(_sort, ascending=[False] + [True] * (len(_sort) - 1))
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
            df = df.rename(columns=_RENAME)
            render_table(df, stats_df=pd.DataFrame())
            download_button(df, "team_hitting_gamelogs_date", "thgl_date_dl")

    with tab_team:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Level", list(_level_map.keys()), index=0, key="thgl_tm_level"
            )
            season = st.multiselect(
                "Season", season_options(team_hitter_gamelogs),
                default=(
                    [season_options(team_hitter_gamelogs)[1]]
                    if len(season_options(team_hitter_gamelogs)) > 1 else ["All"]
                ),
                key="thgl_tm_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(team_hitter_gamelogs),
                index=0, key="thgl_tm_gtg",
            )
            base = team_hitter_gamelogs[
                team_hitter_gamelogs["level_id"].isin(_level_map[level])
            ]
            base = filter_by_values(base, "season", season)
            base = filter_by_game_type_group(base, game_type_group)
            team = st.selectbox(
                "Team", [t for t in team_options(base, "hitting_code") if t != "All"],
                index=0, key="thgl_tm_team",
            )
        with right:
            df = filter_by_team_token(base, "hitting_code", team)
            team_cols = [c for c in _T_H_GL_COLS if c != "hitting_code"]
            df = df[[c for c in team_cols if c in df.columns]].copy()
            df = df.sort_values("game_date", ascending=False) if "game_date" in df.columns else df
            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
            df = df.rename(columns=_RENAME)
            render_table(df, stats_df=pd.DataFrame())
            download_button(df, "team_hitting_gamelogs_team", "thgl_tm_dl")


def team_pitching_gamelogs():
    """Teams - Pitching Game Logs page"""
    st.title("Team Pitching Game Logs")

    if team_pitcher_gamelogs.empty:
        st.info("Missing pitcher_gamelogs.parquet — run the daily pipeline to generate it.")
        return

    _T_P_GL_COLS = [
        "game_date", "pitching_code", "game_pk", "opp_team",
        "TBF", "bbe", "pitches", "whiffs", "chases", "stuff", "grade_v13",
        "HR", "XBH", "hits", "damaged_bbe",
        "la_gte_20_bbe", "la_lte_0_bbe", "BB", "K",
        "strikes", "balls", "swings",
        "zone_pitches", "out_of_zone", "FA", "BR", "OFF", "vs_LHB", "vs_RHB",
    ]
    _RENAME = {
        "game_date": "Date", "pitching_code": "Team",
        "game_pk": "Game ID", "opp_team": "vs",
        "bbe": "BBE", "damaged_bbe": "Damage BBE", "hits": "H",
        "la_gte_20_bbe": "LA >= 20", "la_lte_0_bbe": "LA<=0",
        "swings": "Swings", "chases": "Chases", "whiffs": "Whiffs",
        "pitches": "Pitches",
        "zone_pitches": "Zone", "out_of_zone": "Out of Zone",
        "strikes": "Strikes", "balls": "Balls",
        "FA": "FA#", "BR": "BR#", "OFF": "OFF#",
        "vs_LHB": "vs LHB", "vs_RHB": "vs RHB",
        "stuff": "Pitch Grade", "grade_v13": "Exec Grade",
    }
    _level_map = {
        "MLB": [1], "Triple-A": [11], "Low-A": [14], "Low Minors": [16],
    }

    tab_date, tab_team = st.tabs(["By Date", "By Team"])

    with tab_date:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Level", list(_level_map.keys()), index=0, key="tpgl_date_level"
            )
            season = st.multiselect(
                "Season", season_options(team_pitcher_gamelogs),
                default=(
                    [season_options(team_pitcher_gamelogs)[1]]
                    if len(season_options(team_pitcher_gamelogs)) > 1 else ["All"]
                ),
                key="tpgl_date_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(team_pitcher_gamelogs),
                index=0, key="tpgl_date_gtg",
            )
            base = team_pitcher_gamelogs[
                team_pitcher_gamelogs["level_id"].isin(_level_map[level])
            ]
            base = filter_by_values(base, "season", season)
            base = filter_by_game_type_group(base, game_type_group)
            dates = sorted(base["game_date"].dropna().astype(str).unique(), reverse=True)
            date_choice = st.selectbox(
                "Date", ["All"] + (dates if dates else ["(none)"]), index=0, key="tpgl_date_date",
                format_func=lambda d: (
                    pd.to_datetime(d).strftime("%m/%d/%Y") if d not in ("All", "(none)", "") else d
                ),
            )
            team = st.selectbox(
                "Team", team_options(base, "pitching_code"), index=0, key="tpgl_date_team"
            )
        with right:
            df = base.copy() if date_choice == "All" else base[base["game_date"].astype(str) == date_choice].copy()
            df = filter_by_team_token(df, "pitching_code", team)
            df = df[[c for c in _T_P_GL_COLS if c in df.columns]].copy()
            if "game_date" in df.columns:
                _sort = ["game_date", "pitching_code"] if "pitching_code" in df.columns else ["game_date"]
                df = df.sort_values(_sort, ascending=[False] + [True] * (len(_sort) - 1))
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
            df = df.rename(columns=_RENAME)
            render_table(df, stats_df=pd.DataFrame())
            download_button(df, "team_pitching_gamelogs_date", "tpgl_date_dl")

    with tab_team:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Level", list(_level_map.keys()), index=0, key="tpgl_tm_level"
            )
            season = st.multiselect(
                "Season", season_options(team_pitcher_gamelogs),
                default=(
                    [season_options(team_pitcher_gamelogs)[1]]
                    if len(season_options(team_pitcher_gamelogs)) > 1 else ["All"]
                ),
                key="tpgl_tm_season",
            )
            game_type_group = st.selectbox(
                "Game Type", game_type_group_options(team_pitcher_gamelogs),
                index=0, key="tpgl_tm_gtg",
            )
            base = team_pitcher_gamelogs[
                team_pitcher_gamelogs["level_id"].isin(_level_map[level])
            ]
            base = filter_by_values(base, "season", season)
            base = filter_by_game_type_group(base, game_type_group)
            team = st.selectbox(
                "Team", [t for t in team_options(base, "pitching_code") if t != "All"],
                index=0, key="tpgl_tm_team",
            )
        with right:
            df = filter_by_team_token(base, "pitching_code", team)
            team_cols = [c for c in _T_P_GL_COLS if c != "pitching_code"]
            df = df[[c for c in team_cols if c in df.columns]].copy()
            df = df.sort_values("game_date", ascending=False) if "game_date" in df.columns else df
            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%m/%d/%Y")
            df = df.rename(columns=_RENAME)
            render_table(df, stats_df=pd.DataFrame())
            download_button(df, "team_pitching_gamelogs_team", "tpgl_tm_dl")
