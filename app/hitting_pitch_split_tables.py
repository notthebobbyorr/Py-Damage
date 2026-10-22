"""Standalone team and league hitting pitch-group tables."""
import numpy as np
import pandas as pd
import streamlit as st
from app.filters import apply_workload_maximum

from app.config import LEVEL_LABELS
from app.data_loader import load_csv
from app.filters import (
    download_button, season_options, filter_by_game_type_group, filter_by_team_token,
    player_id_options, team_options,
    _split_team_tokens,
    split_workload_controls,
)
from app.hitter_pitch_splits import COLUMNS, GROUPS
from app.viz import render_table


@st.cache_resource(max_entries=2)
def prepare_hitter_pitch_splits(source, metadata):
    keys = ["batter_mlbid", "season", "level_id"]
    metadata = filter_by_game_type_group(metadata, "Regular Season")
    metadata = metadata[keys + ["hitter_name", "hitting_code"]].copy()
    # Name variants must never multiply pitch-group rows for a player identity.
    metadata = metadata.groupby(keys, observed=True, as_index=False).agg(
        hitter_name=("hitter_name", "first"),
        hitting_code=("hitting_code", lambda s: " | ".join(sorted({
            token for value in s.dropna() for token in _split_team_tokens(str(value))
        }))),
    )
    result = source.merge(metadata, on=keys, how="left", validate="many_to_one")
    result["hitter_name"] = result.hitter_name.astype("string").fillna(result.batter_mlbid.astype(str))
    result["hitting_code"] = result.hitting_code.astype("string").fillna("")
    return result


def render_hitter_pitch_split_leaderboard(metadata):
    prefix = "hitter_pitch_leaderboard"
    source = load_csv("hitter_pitch_splits.parquet")
    if source.empty:
        st.info("Pitch type splits are not available yet.")
        return
    source = prepare_hitter_pitch_splits(source, metadata)
    left, right = st.columns([1, 3])
    with left:
        levels = {"MLB": 1, "Triple-A": 11, "Low-A": 14, "Low Minors": 16}
        level = st.selectbox("Select Level", list(levels) + ["All"], key=prefix + "_level")
        options = season_options(source)
        seasons = st.multiselect("Select Season", options, default=[options[1]], key=prefix + "_season")
        df = source[source.level_id.isin(list(levels.values()) if level == "All" else [levels[level]])].copy()
        if "All" not in seasons:
            df = df[df.season.isin(seasons)]
        team = st.selectbox("Select Team", team_options(df, "hitting_code"), key=prefix + "_team")
        df = filter_by_team_token(df, "hitting_code", team)
        players, names = player_id_options(df, "batter_mlbid", "hitter_name")
        selected = st.multiselect("Select Player", players, default=["All"], key=prefix + "_player",
                                  format_func=lambda v: "All" if v == "All" else f"{names.get(v, 'Unknown')} ({int(v)})")
        if "All" not in selected:
            df = df[df.batter_mlbid.isin(selected)]
        groups = st.multiselect("Select Pitch Type", ["All"] + list(GROUPS.values()),
                                default=["All"], key=prefix + "_pitch_type")
        if "All" not in groups:
            df = df[df.pitch_group.isin([key for key, label in GROUPS.items() if label in groups])]
        workload, minimum = split_workload_controls(source, prefix + "_workload")
        df = df[df[workload] >= minimum]
        df = apply_workload_maximum(df,prefix + "_workload")
    with right:
        st.caption("Regular season. Each row is a hitter's season and level against one pitch group. "
                   "# = pitches seen; % = share of all pitches seen; Whiff% = whiffs / swings. "
                   "Team filters select season affiliations; metrics cover all teams at that level. "
                   "Minimums apply to each pitch-group row.")
        df["Level"] = df.level_id.map(LEVEL_LABELS)
        df["Pitch Type"] = df.pitch_group.map(GROUPS)
        identity = ["hitter_name", "batter_mlbid", "hitting_code", "Level", "season", "Pitch Type"]
        display = df[identity + (["PA"] if "PA" in df else []) + list(COLUMNS)].rename(columns={
            "hitter_name": "Name", "batter_mlbid": "Player ID", "hitting_code": "Team",
            "season": "Season", **COLUMNS,
        }).replace([np.inf, -np.inf], np.nan)
        display = render_table(display, show_maximum_filters=False, conditional_formatting=False, default_sort_col="Damage/BBE (%)", workload_kind="hitter",
                     no_format_cols={"Name", "Player ID", "Team", "Level", "Season", "Pitch Type", "#", "BBE"})
        download_button(display, prefix, prefix + "_download")


def render_hitting_pitch_splits(team=False):
    scope = "team" if team else "league"
    prefix = f"{scope}_hitting_pitch_splits"
    source = load_csv(f"{prefix}.parquet")
    if source.empty:
        st.info("Pitch type splits are not available yet.")
        return
    left, right = st.columns([1, 3])
    with left:
        levels = {"MLB": 1, "Triple-A": 11, "Low-A": 14, "Low Minors": 16}
        level = st.selectbox("Select Level", list(levels) + ["All"], key=prefix + "_level")
        options = season_options(source)
        seasons = st.multiselect("Select Season", options, default=[options[1]], key=prefix + "_season")
        df = source[source.level_id.isin(list(levels.values()) if level == "All" else [levels[level]])].copy()
        if "All" not in seasons:
            df = df[df.season.isin(seasons)]
        if team:
            selected = st.multiselect("Select Team", ["All"] + sorted(df.hitting_code.dropna().unique().tolist()),
                                      default=["All"], key=prefix + "_team")
            if "All" not in selected:
                df = df[df.hitting_code.isin(selected)]
        pitch_types = st.multiselect("Select Pitch Type", ["All"] + list(GROUPS.values()),
                                    default=["All"], key=prefix + "_pitch_type")
        if "All" not in pitch_types:
            df = df[df.pitch_group.isin([group for group, label in GROUPS.items() if label in pitch_types])]
        workload, minimum = split_workload_controls(source, prefix + "_workload")
        df = df[df[workload] >= minimum]
        df = apply_workload_maximum(df,prefix + "_workload")
    with right:
        if df.empty:
            st.info("No pitch type splits for these filters. Team coverage requires named batting-team codes." if team
                    else "No pitch type splits for these filters.")
            return
        st.caption("Regular season. # = pitches seen; % = share of all pitches seen by this "
                   + ("team" if team else "league level")
                   + " in this season. Whiff% = whiffs / swings. Unknown pitch types remain in the % denominator.")
        df["Level"] = df.level_id.map(LEVEL_LABELS)
        df["Pitch Type"] = pd.Categorical(df.pitch_group.map(GROUPS), categories=list(GROUPS.values()), ordered=True)
        identity = (["hitting_code"] if team else []) + ["Level", "season", "Pitch Type"]
        df = df.sort_values((["hitting_code"] if team else []) + ["level_id", "season", "Pitch Type"])
        display = df[identity + (["PA"] if "PA" in df else []) + list(COLUMNS)].rename(columns={"hitting_code": "Team", "season": "Season", **COLUMNS})
        display = display.replace([np.inf, -np.inf], np.nan)
        display = render_table(display, show_maximum_filters=False, conditional_formatting=False, show_controls=False,
                     hide_cols={"PA"} if team else None,
                     no_format_cols={"Team", "Level", "Season", "Pitch Type", "#", "BBE"})
        download_button(display, prefix, prefix + "_download")
