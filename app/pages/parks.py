from __future__ import annotations

import streamlit as st

from app.config import DEFAULT_NO_FORMAT_COLS
from app.datasets import park_data
from app.filters import (
    download_button,
    filter_by_values,
    season_options,
)
from app.viz import render_table


def park_data_page():
    """Parks - HR per Damage BBE page"""
    st.title("Park HR per Damage BBE")

    if park_data.empty:
        st.info("Missing park_data.csv (or park_data.parquet)")
        return

    left, right = st.columns([1, 3])
    with left:
        level = st.selectbox(
            "Select Level",
            ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
            index=1,
            key="park_level",
        )
        season = st.multiselect(
            "Select Season",
            season_options(park_data),
            default=(
                [season_options(park_data)[1]]
                if len(season_options(park_data)) > 1
                else ["All"]
            ),
            key="park_season",
        )
        stands = st.multiselect(
            "Select Batter Handedness",
            ["All"] + sorted(park_data["stands"].dropna().unique().tolist()),
            default=["All"],
            key="park_stands",
        )
        team = st.selectbox(
            "Select Home Team",
            ["All"] + sorted(park_data["home_team"].dropna().unique().tolist()),
            index=0,
            key="park_home_team",
        )
        park_pairs = (
            park_data[["park_mlbid", "home_team"]]
            .dropna()
            .drop_duplicates()
            .sort_values(by=["park_mlbid", "home_team"])
            .values.tolist()
        )
        park_options = ["All"] + [tuple(pair) for pair in park_pairs]
        park = st.selectbox(
            "Select Park",
            park_options,
            index=0,
            key="park_mlbid",
            format_func=lambda v: ("All" if v == "All" else f"{v[0]} - {v[1]}"),
        )
    with right:
        level_map = {
            "All": [1, 11, 14, 16],
            "MLB": [1],
            "Triple-A": [11],
            "Low-A": [14],
            "Low Minors": [16],
        }
        df = park_data.copy()
        df = df[df["level_id"].isin(level_map[level])]
        df = filter_by_values(df, "season", season)
        df = filter_by_values(df, "stands", stands)
        if team != "All":
            df = df[df["home_team"] == team]
        if park != "All":
            df = df[(df["park_mlbid"] == park[0]) & (df["home_team"] == park[1])]
        df = df.assign(
            Level=df["level_id"].map(
                {1: "MLB", 11: "Triple-A", 14: "Low-A", 16: "Low Minors"}
            ),
            __season=df["season"],
            __level=df["level_id"],
        )

        columns = [
            "park_mlbid",
            "home_team",
            "season",
            "stands",
            "Level",
            "damage_bbe",
            "HR_per_damage_BBE_pct",
            "XBH_per_damage_BBE_pct",
            "Hits_per_BBE_pct",
            "__season",
            "__level",
        ]
        df = df[[col for col in columns if col in df.columns]].copy()
        rename_map = {
            "park_mlbid": "Park ID",
            "home_team": "Home Team",
            "stands": "Batter Hand",
            "season": "Season",
            "damage_bbe": "Damage BBE",
            "HR_per_damage_BBE_pct": "HR per Damage BBE (%)",
            "XBH_per_damage_BBE_pct": "XBH per Damage BBE (%)",
            "Hits_per_BBE_pct": "Hits per BBE (%)",
        }
        df = df.rename(columns=rename_map)
        df = df.sort_values(by="HR per Damage BBE (%)", ascending=False)
        render_table(
            df,
            group_cols=["__season", "__level"],
            no_format_cols=DEFAULT_NO_FORMAT_COLS | {"Park ID", "Damage BBE"},
        )
        download_button(df, "park_data", "park_data_download")
