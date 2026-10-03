"""Saved official pitching totals and display formatting."""
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import DATA_DIR

DETAIL_COLUMNS = ["G", "GS", "IP", "W-L", "QS", "ERA", "xFIP", "WHIP", "K%", "BB%", "K-BB%", "AVG", "BABIP", "OPS", "TBF", "SO", "HR", "HA", "SV", "BS", "Hld", "Inh. Runners %"]


@st.cache_resource
def load_official_pitching():
    path = Path(DATA_DIR) / "official_pitching.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path).set_index(["player_id", "season", "level_id"], verify_integrity=True)


def detail_values(row):
    def number(value, decimals=0):
        return "—" if pd.isna(value) else f"{value:.{decimals}f}"

    values = []
    for col in DETAIL_COLUMNS:
        if col == "W-L":
            values.append(f"{number(row['W'])}-{number(row['L'])}")
        elif col == "IP":
            values.append("—" if pd.isna(row[col]) else str(row[col]))
        elif col == "xFIP":
            values.append(number(row.get(col, float("nan")), 2))
        elif col in {"ERA", "WHIP"}:
            values.append(number(row[col], 2))
        elif col in {"AVG", "BABIP", "OPS"}:
            values.append(number(row[col], 3).removeprefix("0"))
        elif col.endswith("%"):
            values.append("—" if pd.isna(row[col]) else number(row[col], 1) + "%")
        else:
            values.append(number(row["H" if col == "Hld" else col]))
    return values
