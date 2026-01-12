from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import colors

DATA_DIR = Path(__file__).resolve().parent
_TABLE_COUNTER = 0

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


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_damage_df() -> pd.DataFrame:
    preferred = DATA_DIR / "damage_pos_2021_2024.csv"
    if preferred.exists():
        return pd.read_csv(preferred)
    candidates = sorted(DATA_DIR.glob("damage_pos_*.csv"))
    if candidates:
        return pd.read_csv(candidates[-1])
    return pd.DataFrame()


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


def render_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available yet.")
        return

    global _TABLE_COUNTER
    table_key = f"table_{_TABLE_COUNTER}"
    _TABLE_COUNTER += 1

    page_size = st.number_input(
        "Rows per page",
        min_value=10,
        max_value=500,
        value=25,
        step=5,
        key=f"{table_key}_page_size",
    )
    total_rows = len(df)
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
    df_page = df.iloc[start:end].copy()

    max_elements = pd.get_option("styler.render.max_elements")
    total_cells = df_page.shape[0] * df_page.shape[1]
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0 and total_cells <= max_elements:
        q10 = df[numeric_cols].quantile(0.05)
        q90 = df[numeric_cols].quantile(0.95)
        med = df[numeric_cols].median()
        cmap = colors.LinearSegmentedColormap.from_list("rwgn", ["#c75c5c", "#f7f7f7", "#5cb85c"])

        def style_column(col: pd.Series) -> list[str]:
            vmin = q10[col.name]
            vmax = q90[col.name]
            vcenter = med[col.name]
            if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
                return [""] * len(col)
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            vals = col.to_numpy(dtype=float)
            vals = np.clip(vals, vmin, vmax)
            return [
                "" if pd.isna(val) else f"background-color: {colors.to_hex(cmap(norm(val)))}"
                for val in vals
            ]

        styler = df_page.style.apply(style_column, subset=numeric_cols, axis=0)
        st.dataframe(styler, width="stretch", hide_index=True)
        return
    st.dataframe(df_page, width="stretch", hide_index=True)


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
hitting_cor = load_csv("hitting_cor.csv")
pitching_cor = load_csv("pitching_cor2.csv")

if not pitch_types.empty and "pitch_group" not in pitch_types.columns and "pitch_tag" in pitch_types.columns:
    pitch_types = pitch_types.assign(
        pitch_group=pitch_types["pitch_tag"].map(
            lambda tag: "FA"
            if tag in {"FA", "HC", "SI"}
            else "BR"
            if tag in {"SL", "SW", "CU"}
            else "OFF"
            if tag in {"CH", "FS"}
            else "OTHER"
        )
    )

st.title("Profiles")

main_tabs = st.tabs(
    [
        "Welcome Page",
        "Hitters",
        "Hitters - Percentiles",
        "Pitchers",
        "Pitchers - Percentiles",
        "Individual Pitches",
        "Individual Pitches - Percentiles",
        "Team Hitting",
        "Team Pitching",
        "League Averages - Hitting",
        "League Averages - Pitching",
        "Glossary - Hitting",
        "Glossary - Pitching",
        "PA & R/RBI Calculator",
        "Hitting Correlations",
        "Pitching Correlations",
    ]
)

with main_tabs[0]:
    st.subheader("Welcome to My App")
    st.markdown(
        """
Here you will find metrics I (https://twitter.com/NotTheBobbyOrr) have developed for analyzing hitters & pitchers at a player and team level.
I make frequent use of these statistics in my work at BaseballProspectus dot com (https://www.baseballprospectus.com/author/ringtheodubel/) and for my own fantasy strategy.
You can navigate via the tabs and there are glossaries containing explanations for each statistic below.

Tip jar if you're feeling generous:
- Venmo: @Robert-Orr7
- Paypal: orrrobf @ gmail dot com

Feedback: If you have any suggestions or just want to say hi, shoot me a DM on Twitter or send me an email at orrrobf @ gmail dot com.
"""
    )
    st.write(f"Last Update: {pd.Timestamp.today().date()}")

with main_tabs[1]:
    st.subheader("Hitters")
    if damage_df.empty:
        st.info("Missing damage_pos_2021_2024.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox("Select Level", ["All", "MLB", "Triple-A", "Low-A", "Low Minors"], index=1)
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(damage_df["season"].dropna().unique().tolist()),
                default=["All"],
            )
            min_value = st.number_input("Minimum Value", min_value=0, max_value=500, value=1, step=1)
            value_type = st.selectbox("Filter By", ["PA", "BBE"], index=1)
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(damage_df["hitting_code"].dropna().unique().tolist()),
                default=["All"],
            )
            player = st.multiselect(
                "Select Player",
                ["All"] + sorted(damage_df["hitter_name"].dropna().unique().tolist()),
                default=["All"],
            )
            positions = st.multiselect("Select Positions", ["C", "1B", "2B", "SS", "3B", "OF", "UT", "P"], default=[])
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = damage_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_values(df, "hitting_code", team)
            df = filter_by_values(df, "hitter_name", player)

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            pos_map = {"C": "C", "1B": "X1B", "2B": "X2B", "3B": "X3B", "SS": "SS", "OF": "OF", "UT": "UT", "P": "P"}
            for pos in positions:
                col = pos_map.get(pos)
                if col and col in df.columns:
                    df = df[df[col] >= 1]

            df = df[
                [
                    "hitter_name",
                    "hitting_code",
                    "season",
                    "PA",
                    "bbe",
                    "damage_rate",
                    "EV90th",
                    "max_EV",
                    "pull_FB_pct",
                    "SEAGER",
                    "selection_skill",
                    "hittable_pitches_taken",
                    "chase",
                    "z_con",
                    "secondary_whiff_pct",
                    "contact_vs_avg",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "hitter_name": "Name",
                    "hitting_code": "Team",
                    "season": "Season",
                    "bbe": "BBE",
                    "damage_rate": "Damage/BBE (%)",
                    "EV90th": "90th Pctile EV",
                    "max_EV": "Max EV",
                    "pull_FB_pct": "Pulled FB (%)",
                    "selection_skill": "Selectivity (%)",
                    "hittable_pitches_taken": "Hittable Pitch Take (%)",
                    "chase": "Chase (%)",
                    "z_con": "Z-Contact (%)",
                    "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                    "contact_vs_avg": "Contact Over Expected (%)",
                }
            )
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            render_table(df)
            download_button(df, "hitters", "hitters_download")

with main_tabs[2]:
    st.subheader("Percentile Rankings - Hitters")
    st.caption("min. 100 pitches seen & 20 batted balls at respective level")
    if hitter_pct.empty:
        st.info("Missing hitter_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="hit_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(hitter_pct["season"].dropna().unique().tolist()),
                default=["All"],
                key="hit_pct_season",
            )
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(hitter_pct["hitting_code"].dropna().unique().tolist()),
                default=["All"],
                key="hit_pct_team",
            )
            player = st.multiselect(
                "Select Player",
                ["All"] + sorted(hitter_pct["hitter_name"].dropna().unique().tolist()),
                default=["All"],
                key="hit_pct_player",
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
            df = df[
                [
                    "hitter_name",
                    "season",
                    "hitting_code",
                    "SEAGER_pctile",
                    "selection_pctile",
                    "hittable_pitches_pctile",
                    "damage_pctile",
                    "EV90_pctile",
                    "max_pctile",
                    "pfb_pctile",
                    "chase_pctile",
                    "z_con_pctile",
                    "sec_whiff_pctile",
                    "c_vs_avg_pctile",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "hitter_name": "Name",
                    "hitting_code": "Team",
                    "season": "Season",
                    "SEAGER_pctile": "SEAGER",
                    "selection_pctile": "Selection Skill",
                    "hittable_pitches_pctile": "Hittable Pitch Take",
                    "damage_pctile": "Damage Rate",
                    "EV90_pctile": "90th Pctile EV",
                    "max_pctile": "Max EV",
                    "pfb_pctile": "Pulled FB",
                    "chase_pctile": "Chase",
                    "z_con_pctile": "Z-Contact",
                    "sec_whiff_pctile": "Whiff vs Secondaries",
                    "c_vs_avg_pctile": "Contact Over Expected",
                }
            )
            df = df.sort_values(by="SEAGER", ascending=False)
            render_table(df)
            download_button(df, "hitter_percentiles", "hitter_pct_download")

with main_tabs[3]:
    st.subheader("Pitchers")
    if pitcher_df.empty:
        st.info("Missing pitcher_stuff_new.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox("Select Level", ["All", "MLB", "Triple-A", "Low-A", "Low Minors"], index=1, key="pit_level")
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(pitcher_df["season"].dropna().unique().tolist()),
                default=["All"],
                key="pit_season",
            )
            min_value = st.number_input("Minimum Value", min_value=0, max_value=1000, value=10, step=1, key="pit_min")
            filter_type = st.selectbox("Filter By", ["IP", "TBF"], index=1, key="pit_filter")
            per_game_min = st.number_input(
                "Min TBF per Game", min_value=0, max_value=30, value=0, step=1, key="pit_tbf_per_game_min"
            )
            per_game_max = st.number_input(
                "Max TBF per Game", min_value=0, max_value=30, value=30, step=1, key="pit_tbf_per_game_max"
            )
            hand = st.selectbox("Select Pitcher Hand", ["Both", "LHP", "RHP"], key="pit_hand")
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(pitcher_df["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pit_team",
            )
            player = st.multiselect(
                "Select Pitcher",
                ["All"] + sorted(pitcher_df["name"].dropna().unique().tolist()),
                default=["All"],
                key="pit_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            hand_map = {"Both": ["L", "R"], "LHP": ["L"], "RHP": ["R"]}
            df = pitcher_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = df[df["pitcher_hand"].isin(hand_map[hand])]
            df = filter_by_values(df, "season", season)
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", player)
            df = df[(df["TBF_per_G"] >= per_game_min) & (df["TBF_per_G"] <= per_game_max)]
            if filter_type == "IP":
                df = numeric_filter(df, "IP", min_value)
            else:
                df = numeric_filter(df, "TBF", min_value)
            df = df[
                [
                    "name",
                    "season",
                    "pitching_code",
                    "TBF",
                    "IP",
                    "std.ZQ",
                    "std.DMG",
                    "std.NRV",
                    "fastball_velo",
                    "max_velo",
                    "fastball_vaa",
                    "SwStr",
                    "Ball_pct",
                    "Z_Contact",
                    "Chase",
                    "CSW",
                    "rel_z",
                    "rel_x",
                    "ext",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "name": "Name",
                    "pitching_code": "Team",
                    "season": "Season",
                    "std.ZQ": "Pitch Quality",
                    "std.DMG": "Damage Suppression",
                    "std.NRV": "Non-BIP Skill",
                    "fastball_velo": "FA mph",
                    "max_velo": "Max FA mph",
                    "fastball_vaa": "FA VAA",
                    "SwStr": "SwStr (%)",
                    "Ball_pct": "Ball (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "CSW": "CSW (%)",
                    "rel_z": "Vertical Release (ft.)",
                    "rel_x": "Horizontal Release (ft.)",
                    "ext": "Extension (ft.)",
                }
            )
            df = df.sort_values(by="Pitch Quality", ascending=False)
            render_table(df)
            download_button(df, "pitchers", "pitchers_download")

with main_tabs[4]:
    st.subheader("Percentile Rankings - Pitchers")
    st.caption("min. 100 pitches thrown at respective level")
    if pitcher_pct.empty:
        st.info("Missing pitcher_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pit_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(pitcher_pct["season"].dropna().unique().tolist()),
                default=["All"],
                key="pit_pct_season",
            )
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(pitcher_pct["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pit_pct_team",
            )
            player = st.multiselect(
                "Select Pitcher",
                ["All"] + sorted(pitcher_pct["name"].dropna().unique().tolist()),
                default=["All"],
                key="pit_pct_player",
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
            df = df[
                [
                    "name",
                    "season",
                    "pitching_code",
                    "PQ_pctile",
                    "DMG_pctile",
                    "NRV_pctile",
                    "FA_velo_pctile",
                    "FA_max_pctile",
                    "FA_vaa_pctile",
                    "SwStr_pctile",
                    "Ball_pctile",
                    "Z_con_pctile",
                    "Chase_pctile",
                    "CSW_pctile",
                    "rZ_pctile",
                    "rX_pctile",
                    "ext_pctile",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "name": "Name",
                    "pitching_code": "Team",
                    "season": "Season",
                    "PQ_pctile": "Pitch Quality",
                    "DMG_pctile": "Damage Suppression",
                    "NRV_pctile": "Non-BIP Skill",
                    "FA_velo_pctile": "Avg FA mph",
                    "FA_max_pctile": "Max FA mph",
                    "FA_vaa_pctile": "FA VAA",
                    "SwStr_pctile": "SwStr (%)",
                    "Ball_pctile": "Ball (%)",
                    "Z_con_pctile": "Z-Contact (%)",
                    "Chase_pctile": "Chase (%)",
                    "CSW_pctile": "CSW (%)",
                    "rZ_pctile": "Vertical Release (ft.)",
                    "rX_pctile": "Horizontal Release (ft.)",
                    "ext_pctile": "Extension (ft.)",
                }
            )
            df = df.sort_values(by="Pitch Quality", ascending=False)
            render_table(df)
            download_button(df, "pitcher_percentiles", "pitcher_pct_download")

with main_tabs[5]:
    st.subheader("Individual Pitches")
    if pitch_types.empty:
        st.info("Missing new_pitch_types.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pt_level",
            )
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(pitch_types["season"].dropna().unique().tolist()),
                default=["All"],
                key="pt_season",
            )
            hand = st.selectbox("Select Pitcher Hand", ["Both", "LHP", "RHP"], key="pt_hand")
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(pitch_types["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pt_team",
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                ["All"] + sorted(pitch_types["name"].dropna().unique().tolist()),
                default=["All"],
                key="pt_pitcher",
            )
            pitch_group = st.multiselect(
                "Select Pitch Group",
                ["All"] + sorted(pitch_types["pitch_group"].dropna().unique().tolist()),
                default=["All"],
                key="pt_group",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"] + sorted(pitch_types["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pt_tag",
            )
            min_pitches = st.number_input(
                "Minimum Pitches", min_value=0, max_value=3000, value=10, step=1, key="pt_min"
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            hand_map = {"Both": ["L", "R"], "LHP": ["L"], "RHP": ["R"]}
            df = pitch_types.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = df[df["pitcher_hand"].isin(hand_map[hand])]
            df = filter_by_values(df, "season", season)
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", pitcher)
            df = filter_by_values(df, "pitch_group", pitch_group)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]
            df = df[
                [
                    "name",
                    "pitching_code",
                    "season",
                    "pitch_tag",
                    "pitches",
                    "pct",
                    "std.ZQ",
                    "std.DMG",
                    "std.NRV",
                    "velo",
                    "max_velo",
                    "vaa",
                    "haa",
                    "ivb",
                    "hb",
                    "SwStr",
                    "Z_Contact",
                    "Ball_pct",
                    "Zone",
                    "Chase",
                    "CSW",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "name": "Name",
                    "pitching_code": "Team",
                    "season": "Season",
                    "pitch_tag": "Pitch Type",
                    "pitches": "#",
                    "pct": "Usage (%)",
                    "std.ZQ": "Pitch Quality",
                    "std.NRV": "Non-BIP Skill",
                    "std.DMG": "Damage Suppression",
                    "velo": "Velo",
                    "max_velo": "Max Velo",
                    "vaa": "VAA",
                    "haa": "HAA",
                    "ivb": "IVB (in.)",
                    "hb": "HB (in.)",
                    "CSW": "CSW (%)",
                    "SwStr": "SwStr (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "Zone": "Zone (%)",
                    "Ball_pct": "Ball (%)",
                }
            )
            df = df.sort_values(by="Pitch Quality", ascending=False)
            render_table(df)
            download_button(df, "pitch_types", "pitch_types_download")

with main_tabs[6]:
    st.subheader("Percentile Rankings - Pitch Types")
    st.caption("min. 50 pitches thrown. Percentiles are within pitch type at respective level.")
    if pitch_types_pct.empty:
        st.info("Missing pitch_types_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pt_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(pitch_types_pct["season"].dropna().unique().tolist()),
                default=["All"],
                key="pt_pct_season",
            )
            hand = st.selectbox("Select Pitcher Hand", ["Both", "LHP", "RHP"], key="pt_pct_hand")
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(pitch_types_pct["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="pt_pct_team",
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                ["All"] + sorted(pitch_types_pct["name"].dropna().unique().tolist()),
                default=["All"],
                key="pt_pct_pitcher",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"] + sorted(pitch_types_pct["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pt_pct_tag",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            hand_map = {"Both": ["L", "R"], "LHP": ["L"], "RHP": ["R"]}
            df = pitch_types_pct.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = df[df["pitcher_hand"].isin(hand_map[hand])]
            df = filter_by_values(df, "season", season)
            df = filter_by_values(df, "pitching_code", team)
            df = filter_by_values(df, "name", pitcher)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[
                [
                    "name",
                    "pitching_code",
                    "season",
                    "pitch_tag",
                    "usage_pctile",
                    "PQ_pctile",
                    "DMG_pctile",
                    "NRV_pctile",
                    "velo_pctile",
                    "max_velo_pctile",
                    "vaa_pctile",
                    "haa_pctile",
                    "ivb_pctile",
                    "hb_pctile",
                    "SwStr_pctile",
                    "Ball_pctile",
                    "zone_pctile",
                    "Z_con_pctile",
                    "Chase_pctile",
                    "CSW_pctile",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "name": "Name",
                    "pitching_code": "Team",
                    "season": "Season",
                    "pitch_tag": "Pitch Type",
                    "usage_pctile": "Usage (%)",
                    "PQ_pctile": "Pitch Quality",
                    "NRV_pctile": "Non-BIP Skill",
                    "DMG_pctile": "Damage Suppression",
                    "velo_pctile": "Velo",
                    "max_velo_pctile": "Max Velo",
                    "vaa_pctile": "VAA",
                    "haa_pctile": "HAA",
                    "ivb_pctile": "IVB (in.)",
                    "hb_pctile": "HB (in.)",
                    "CSW_pctile": "CSW (%)",
                    "SwStr_pctile": "SwStr (%)",
                    "Z_con_pctile": "Z-Contact (%)",
                    "Chase_pctile": "Chase (%)",
                    "zone_pctile": "Zone (%)",
                    "Ball_pctile": "Ball (%)",
                }
            )
            df = df.sort_values(by="Pitch Quality", ascending=False)
            render_table(df)
            download_button(df, "pitch_types_percentiles", "pitch_types_pct_download")

with main_tabs[7]:
    st.subheader("Team Hitting")
    if team_damage.empty:
        st.info("Missing new_team_damage.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox("Select Level", ["MLB", "Triple-A", "Low-A", "Low Minors"], index=0, key="team_hit_level")
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(team_damage["season"].dropna().unique().tolist()),
                default=["All"],
                key="team_hit_season",
            )
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(team_damage["hitting_code"].dropna().unique().tolist()),
                default=["All"],
                key="team_hit_team",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = team_damage.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_values(df, "hitting_code", team)
            df = df[
                [
                    "hitting_code",
                    "season",
                    "PA",
                    "bbe",
                    "damage_rate",
                    "EV90th",
                    "pull_FB_pct",
                    "SEAGER",
                    "selection_skill",
                    "hittable_pitches_taken",
                    "chase",
                    "z_con",
                    "secondary_whiff_pct",
                    "contact_vs_avg",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "hitting_code": "Team",
                    "season": "Season",
                    "bbe": "BBE",
                    "damage_rate": "Damage/BBE (%)",
                    "EV90th": "90th Pctile EV",
                    "pull_FB_pct": "Pulled FB (%)",
                    "selection_skill": "Selectivity (%)",
                    "hittable_pitches_taken": "Hittable Pitch Take (%)",
                    "chase": "Chase (%)",
                    "z_con": "Z-Contact (%)",
                    "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                    "contact_vs_avg": "Contact Over Expected (%)",
                }
            )
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            render_table(df)
            download_button(df, "team_hitting", "team_hitting_download")

with main_tabs[8]:
    st.subheader("Team Pitching")
    if team_stuff.empty:
        st.info("Missing new_team_stuff.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["MLB", "Triple-A", "Low-A", "Low Minors"],
                index=0,
                key="team_pitch_level",
            )
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(team_stuff["season"].dropna().unique().tolist()),
                default=["All"],
                key="team_pitch_season",
            )
            team = st.multiselect(
                "Select Team",
                ["All"] + sorted(team_stuff["pitching_code"].dropna().unique().tolist()),
                default=["All"],
                key="team_pitch_team",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = team_stuff.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_values(df, "pitching_code", team)
            df = df[
                [
                    "pitching_code",
                    "season",
                    "IP",
                    "std.ZQ",
                    "std.DMG",
                    "std.NRV",
                    "fastball_velo",
                    "fastball_vaa",
                    "SwStr",
                    "Ball_pct",
                    "Z_Contact",
                    "Chase",
                    "CSW",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "pitching_code": "Team",
                    "season": "Season",
                    "std.ZQ": "Pitch Quality",
                    "std.NRV": "Non-BIP Skill",
                    "std.DMG": "Damage Suppression",
                    "SwStr": "SwStr (%)",
                    "Ball_pct": "Ball (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "CSW": "CSW (%)",
                    "fastball_velo": "FA mph",
                    "fastball_vaa": "FA VAA",
                }
            )
            df = df.sort_values(by="Pitch Quality", ascending=False)
            render_table(df)
            download_button(df, "team_pitching", "team_pitching_download")

with main_tabs[9]:
    st.subheader("League Averages - Hitting")
    if hitting_avg.empty:
        st.info("Missing new_hitting_lg_avg.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            season = st.multiselect(
                "Select Season",
                ["All"] + sorted(hitting_avg["season"].dropna().unique().tolist()),
                default=["All"],
                key="lg_hit_season",
            )
        with right:
            df = hitting_avg.copy()
            df = filter_by_values(df, "season", season)
            df = df.assign(
                Level=df["level_id"].map({1: "MLB", 11: "Triple-A", 14: "Low-A", 16: "Low Minors"})
            )
            df = df[
                [
                    "Level",
                    "season",
                    "PA",
                    "bbe",
                    "damage_rate",
                    "EV90th",
                    "pull_FB_pct",
                    "SEAGER",
                    "selection_skill",
                    "hittable_pitches_taken",
                    "chase",
                    "z_con",
                    "contact_vs_avg",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "season": "Season",
                    "bbe": "BBE",
                    "damage_rate": "Damage/BBE (%)",
                    "EV90th": "90th Pctile EV",
                    "pull_FB_pct": "Pulled FB (%)",
                    "selection_skill": "Selectivity (%)",
                    "hittable_pitches_taken": "Hittable Pitch Take (%)",
                    "chase": "Chase (%)",
                    "z_con": "Z-Contact (%)",
                    "contact_vs_avg": "Contact Over Expected (%)",
                }
            )
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            render_table(df)
            download_button(df, "league_hitting", "league_hitting_download")

with main_tabs[10]:
    st.subheader("League Averages - Pitching")
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
                ["All"] + sorted(pitching_avg["season"].dropna().unique().tolist()),
                default=["All"],
                key="lg_pitch_season",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = pitching_avg.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = df[
                [
                    "season",
                    "fastball_velo",
                    "fastball_vaa",
                    "SwStr",
                    "Ball_pct",
                    "Z_Contact",
                    "Chase",
                    "CSW",
                ]
            ].copy()
            df = df.rename(
                columns={
                    "season": "Season",
                    "CSW": "CSW (%)",
                    "Ball_pct": "Ball (%)",
                    "SwStr": "SwStr (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "fastball_velo": "FA mph",
                    "fastball_vaa": "FA VAA",
                }
            )
            render_table(df)
            download_button(df, "league_pitching", "league_pitching_download")

with main_tabs[11]:
    st.subheader("Hitting Metrics")
    st.markdown(
        """
**Damage** - A batted ball that clears a threshold of exit velocity, launch angle, and hit direction likely to produce an XBH.
Tracked per batted ball.

**Pulled FB (%)** - The percentage of a player's batted balls hit at a launch angle above 20 degrees and a spray angle of 15 degrees or greater to their pull side.

**SEAGER** - Selective Aggression Engagement Rate. The difference between Selection Tendency and Hittable Pitches Taken.

**Selection Tendency** - How many of a player's good decisions (positive expected value) were a result of taking pitches.

**Hittable Pitches Taken** - How many of a player's takes were pitches with a positive expected value.

**Whiff vs. Secondaries** - The whiff per swing rate of a player against breaking & offspeed pitches.

**Contact Over Expected** - A player's contact rate compared to their expected contact rate given the quality of the pitches they swing at.
"""
    )

with main_tabs[12]:
    st.subheader("Pitching Metrics")
    st.markdown(
        """
**Pitch Quality** - A blend of Non-BIP Skill and Damage Suppression, each weighted by that pitcher's tendency to allow or avoid balls in play.

**Non-BIP Skill** - Expected run values of outcomes on non-balls in play based on pitch traits only.

**Damage Suppression** - Ability to avoid damage on balls in play based on pitch traits only.

Pitch model features include Velocity, IVB, HB, VAA, HAA, release angles, spin efficiency, spin axis, RPM,
release height/width, extension, and handedness.
"""
    )

with main_tabs[13]:
    st.subheader("PA & R/RBI Calculator")
    st.info(
        "This calculator relies on RDS models (PA_estimate.rds, runs_per_pa.rds, rbi_per_pa.rds).\n"
        "Convert those models to Python or provide a CSV-based formula to enable it here."
    )

with main_tabs[14]:
    st.subheader("Hitting Correlations")
    st.caption(
        "Weighted correlations of key metrics on current and next season statistics among hitters with 200+ PA in consecutive seasons. Data from 2021-2023 seasons."
    )
    render_table(hitting_cor)
    download_button(hitting_cor, "hitting_correlations", "hit_cor_download")

with main_tabs[15]:
    st.subheader("Pitching Correlations")
    st.caption(
        "Weighted correlations of key metrics on current and next season statistics among pitchers with 40+ IP in consecutive seasons. Data from 2021-2023 seasons."
    )
    render_table(pitching_cor)
    download_button(pitching_cor, "pitching_correlations", "pitch_cor_download")
