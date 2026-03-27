from __future__ import annotations

import pandas as pd

from app.data_loader import load_csv, load_damage_df
from app.utils import (
    _build_hitter_mlb_equivalencies,
    _build_pitcher_mlb_equivalencies,
    _merge_regressed,
    _normalize_la_cols,
    _normalize_split_cols,
    _normalize_team_col,
)

# ---------------------------------------------------------------------------
# Raw loads
# ---------------------------------------------------------------------------
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
hitters_regressed = load_csv("hitters_regressed.csv")
pitchers_regressed = load_csv("pitchers_regressed.csv")
pitch_types_regressed = load_csv("pitch_types_regressed.csv")
hitter_splits_df = load_csv("hitter_splits.csv")
pitcher_splits_df = load_csv("pitcher_splits.csv")
pitch_type_splits_df = load_csv("pitch_types_splits.csv")
league_pitch_types = load_csv("league_pitch_types.csv")
park_data = load_csv("park_data.csv")

# ---------------------------------------------------------------------------
# Normalize column names
# ---------------------------------------------------------------------------
damage_df = _normalize_team_col(damage_df, "hitting_code")
damage_df = _normalize_la_cols(damage_df)
hitter_pct = _normalize_team_col(hitter_pct, "hitting_code")
hitter_pct = _normalize_la_cols(hitter_pct)
hitter_splits_df = _normalize_team_col(hitter_splits_df, "hitting_code")
hitter_splits_df = _normalize_la_cols(hitter_splits_df)
hitter_splits_df = _normalize_split_cols(hitter_splits_df)
pitcher_df = _normalize_team_col(pitcher_df, "pitching_code")
pitcher_df = _normalize_la_cols(pitcher_df)
pitcher_pct = _normalize_team_col(pitcher_pct, "pitching_code")
pitcher_splits_df = _normalize_team_col(pitcher_splits_df, "pitching_code")
pitcher_splits_df = _normalize_la_cols(pitcher_splits_df)
pitcher_splits_df = _normalize_split_cols(pitcher_splits_df)
pitch_types = _normalize_team_col(pitch_types, "pitching_code")
pitch_types_pct = _normalize_team_col(pitch_types_pct, "pitching_code")
pitch_type_splits_df = _normalize_team_col(pitch_type_splits_df, "pitching_code")
pitch_type_splits_df = _normalize_split_cols(pitch_type_splits_df)
league_pitch_types = _normalize_split_cols(league_pitch_types)
team_damage = _normalize_la_cols(team_damage)
team_stuff = _normalize_la_cols(team_stuff)

# ---------------------------------------------------------------------------
# Backfill pitch_group if missing
# ---------------------------------------------------------------------------
if (
    not pitch_types.empty
    and "pitch_group" not in pitch_types.columns
    and "pitch_tag" in pitch_types.columns
):
    pitch_types = pitch_types.assign(
        pitch_group=pitch_types["pitch_tag"].map(
            lambda tag: (
                "FA"
                if tag in {"FA", "HC", "SI"}
                else (
                    "BR"
                    if tag in {"SL", "SW", "CU"}
                    else "OFF" if tag in {"CH", "FS"} else "OTHER"
                )
            )
        )
    )

if (
    not pitch_type_splits_df.empty
    and "pitch_group" not in pitch_type_splits_df.columns
    and "pitch_tag" in pitch_type_splits_df.columns
):
    pitch_type_splits_df = pitch_type_splits_df.assign(
        pitch_group=pitch_type_splits_df["pitch_tag"].map(
            lambda tag: (
                "FA"
                if tag in {"FA", "HC", "SI"}
                else (
                    "BR"
                    if tag in {"SL", "SW", "CU"}
                    else "OFF" if tag in {"CH", "FS"} else "OTHER"
                )
            )
        )
    )

# ---------------------------------------------------------------------------
# Merge regressed columns
# ---------------------------------------------------------------------------
hitters_reg_df = _merge_regressed(
    damage_df,
    hitters_regressed,
    ["batter_mlbid", "hitter_name", "season", "level_id"],
)
pitchers_reg_df = _merge_regressed(
    pitcher_df,
    pitchers_regressed,
    ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand"],
)
pitch_types_reg_df = _merge_regressed(
    pitch_types,
    pitch_types_regressed,
    ["pitcher_mlbid", "name", "pitcher_hand", "season", "level_id", "pitch_tag"],
)

# ---------------------------------------------------------------------------
# MLB equivalency tables
# ---------------------------------------------------------------------------
hitters_mlb_eq_df, hitter_mlb_eq_coeffs, hitter_mlb_eq_metrics = (
    _build_hitter_mlb_equivalencies(hitters_reg_df)
)
pitchers_mlb_eq_df, pitcher_mlb_eq_coeffs, pitcher_mlb_eq_metrics = (
    _build_pitcher_mlb_equivalencies(pitchers_reg_df)
)
