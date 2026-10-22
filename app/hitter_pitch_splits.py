"""Plain hitter pitch-group detail rows."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.config import DATA_DIR
from app.table_layout import order_stat_columns

GROUPS = {"FA": "Fastball", "SI": "Sinker", "BR": "Breaking", "OFF": "Offspeed"}
COLUMNS = {
    "pitches": "#", "pitch_pct": "%", "bbe": "BBE", "HR": "HR",
    "damage_rate": "Damage/BBE (%)", "EV90th": "90th Pctile EV",
    "pull_FB_pct": "Pulled FB (%)", "LA_gte_20": "LA>=20%", "LA_lte_0": "LA<=0%",
    "SEAGER": "SEAGER", "selection_skill": "Selectivity (%)",
    "hittable_pitches_taken": "Hittable Pitch Take (%)", "chase": "Chase (%)",
    "z_con": "Z-Contact (%)", "whiff_pct": "Whiff%",
    "contact_vs_avg": "Contact Over Expected (%)", "Swing_pct": "Swing (%)", "SwStr": "SwStr (%)",
    "bat_speed": "Avg Swing Speed", "fast_swing_pct": "Fast Swing (%)",
    "swing_length": "Swing Length", "swing_path_tilt": "VBA",
    "attack_angle": "Attack Angle", "attack_direction": "Attack Direction",
    "intercept_x_inches": "Intercept X (in.)", "intercept_y_inches": "Intercept Y (in.)",
    "miss_distance": "Avg Miss Distance (in.)",
}
_keys_by_label = {label: key for key, label in COLUMNS.items()}
COLUMNS = {_keys_by_label[label]: label for label in order_stat_columns(COLUMNS.values())}

BAT_TRACKING_COLUMNS = {key: COLUMNS[key] for key in (
    "bat_speed", "fast_swing_pct", "swing_length", "swing_path_tilt",
    "attack_angle", "attack_direction", "intercept_x_inches", "intercept_y_inches",
    "miss_distance",
)}


@st.cache_resource
def load_hitter_pitch_splits():
    path = Path(DATA_DIR) / "hitter_pitch_splits.parquet"
    if not path.exists():
        return pd.DataFrame()
    from app.data_loader import _read_parquet
    from app.swing_rates import add_swing_rates
    return add_swing_rates(_read_parquet(path)).set_index(
        ["batter_mlbid", "season", "level_id", "pitch_group"], verify_integrity=True
    )


def split_values(data, identity):
    if data.empty or not any((*identity, group) in data.index for group in GROUPS):
        return None
    rows = []
    for group, label in GROUPS.items():
        key = (*identity, group)
        source = data.loc[key] if key in data.index else {}
        values = [label]
        for column in COLUMNS:
            value = source.get(column, 0 if column in {"pitches", "pitch_pct", "bbe", "HR"} else np.nan)
            decimals = 0 if column in {"pitches", "bbe", "HR"} else 1
            values.append(f"{value:.{decimals}f}" if pd.notna(value) and np.isfinite(value) else "—")
        rows.append(values)
    return rows
