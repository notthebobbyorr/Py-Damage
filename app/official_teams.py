"""Cached team totals using the shared player detail formats."""
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import DATA_DIR
from app import official_hitting, official_pitching


@st.cache_resource
def load_official_teams(pitching=False):
    path = Path(DATA_DIR) / "official_teams.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame = frame[frame["group"] == ("pitching" if pitching else "hitting")].copy()
    # The app can retain either abbreviation around the Athletics relocation.
    frame["team_code"] = frame.team_code.replace({"OAK": "ATH", "AZ": "ARI"})
    # Some historical minor-league clubs share abbreviations. Never guess.
    frame = frame[~frame.duplicated(["team_code", "season", "level_id"], keep=False)]
    return frame.set_index(["team_code", "season", "level_id"], verify_integrity=True)


def detail_columns(pitching=False):
    columns = list(official_pitching.DETAIL_COLUMNS if pitching else official_hitting.DETAIL_COLUMNS)
    columns.remove("xFIP" if pitching else "wRC+")
    if pitching:
        columns.remove("Inh. Runners %")
        columns.insert(columns.index("ERA") + 1, "RA9")
    else:
        columns.remove("R")
        index = columns.index("HR") + 1
        columns[index:index] = ["R", "R/G"]
    return columns


def detail_values(row, pitching=False):
    module = official_pitching if pitching else official_hitting
    values = dict(zip(module.DETAIL_COLUMNS, module.detail_values(row)))
    col = "RA9" if pitching else "R/G"
    values[col] = "—" if pd.isna(row[col]) else f"{row[col]:.2f}"
    return [values[column] for column in detail_columns(pitching)]
