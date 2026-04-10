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
pitcher_baserunning_reg = load_csv("pitcher_baserunning_regressed.parquet")
pitch_types_regressed = load_csv("pitch_types_regressed.csv")
hitter_splits_df = load_csv("hitter_splits.csv")
pitcher_splits_df = load_csv("pitcher_splits.csv")
pitch_type_splits_df = load_csv("pitch_types_splits.csv")
league_pitch_types = load_csv("league_pitch_types.csv")
park_data = load_csv("park_data.csv")
baserunning_reg = load_csv("baserunning_regressed.parquet")

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


def _recode_team(series: pd.Series, old: str, new: str) -> pd.Series:
    """Replace a team code, preserving CategoricalDtype if present."""
    is_cat = hasattr(series, "cat")
    result = series.astype(str).replace(old, new)
    return result.astype("category") if is_cat else result


team_damage["hitting_code"] = _recode_team(team_damage["hitting_code"], "AZ", "ARI")
team_stuff["pitching_code"] = _recode_team(team_stuff["pitching_code"], "AZ", "ARI")

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
# Merge baserunning columns onto damage_df
# Provides SBO, takeoff_rate (raw), and takeoff_rate_reg for both hitter pages.
# ---------------------------------------------------------------------------
if not baserunning_reg.empty and not damage_df.empty and "game_type_group" in damage_df.columns:
    _br = baserunning_reg.rename(columns={"runner_mlbid": "batter_mlbid"})
    _br_want = ["batter_mlbid", "season", "level_id", "game_type_group",
                "SB", "takeoff_rate_n", "takeoff_rate_raw", "takeoff_rate_reg"]
    _br = (
        _br[[c for c in _br_want if c in _br.columns]]
        .rename(columns={"takeoff_rate_n": "SBO", "takeoff_rate_raw": "takeoff_rate"})
        .drop_duplicates(subset=["batter_mlbid", "season", "level_id", "game_type_group"])
    )
    del _br_want
    _left = damage_df.copy()
    _merge_keys = ["batter_mlbid", "season", "level_id", "game_type_group"]
    for _col in ["batter_mlbid", "season", "level_id"]:
        _left[_col] = pd.to_numeric(_left[_col], errors="coerce").astype("Int64")
        _br[_col] = pd.to_numeric(_br[_col], errors="coerce").astype("Int64")
    damage_df = _left.merge(_br, on=_merge_keys, how="left")
    # Cast integer counting stats to nullable Int64 so they display without trailing .0
    for _int_col in ["SBO", "SB"]:
        if _int_col in damage_df.columns:
            damage_df[_int_col] = pd.to_numeric(damage_df[_int_col], errors="coerce").astype("Int64")
    del _br, _left, _merge_keys, _col, _int_col

# ---------------------------------------------------------------------------
# Merge pitcher baserunning columns onto pitcher_df
# Provides SBO, takeoff_rate (raw), and takeoff_rate_reg for pitcher pages.
# ---------------------------------------------------------------------------
if not pitcher_baserunning_reg.empty and not pitcher_df.empty and "game_type_group" in pitcher_df.columns:
    _pbr = pitcher_baserunning_reg.copy()
    _pbr_want = ["pitcher_mlbid", "season", "level_id", "game_type_group",
                 "SB", "takeoff_rate_n", "takeoff_rate_raw", "takeoff_rate_reg"]
    _pbr = (
        _pbr[[c for c in _pbr_want if c in _pbr.columns]]
        .rename(columns={"takeoff_rate_n": "SBO", "takeoff_rate_raw": "takeoff_rate"})
        .drop_duplicates(subset=["pitcher_mlbid", "season", "level_id", "game_type_group"])
    )
    del _pbr_want
    _pleft = pitcher_df.copy()
    _p_merge_keys = ["pitcher_mlbid", "season", "level_id", "game_type_group"]
    for _pcol in ["pitcher_mlbid", "season", "level_id"]:
        _pleft[_pcol] = pd.to_numeric(_pleft[_pcol], errors="coerce").astype("Int64")
        _pbr[_pcol] = pd.to_numeric(_pbr[_pcol], errors="coerce").astype("Int64")
    pitcher_df = _pleft.merge(_pbr, on=_p_merge_keys, how="left")
    for _p_int_col in ["SBO", "SB"]:
        if _p_int_col in pitcher_df.columns:
            pitcher_df[_p_int_col] = pd.to_numeric(pitcher_df[_p_int_col], errors="coerce").astype("Int64")
    del _pbr, _pleft, _p_merge_keys, _pcol, _p_int_col

# ---------------------------------------------------------------------------
# Merge regressed columns
# ---------------------------------------------------------------------------
hitters_reg_df = _merge_regressed(
    damage_df,
    hitters_regressed,
    ["batter_mlbid", "hitter_name", "season", "level_id", "game_type_group"],
)
pitchers_reg_df = _merge_regressed(
    pitcher_df,
    pitchers_regressed,
    ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand", "game_type_group"],
)
pitch_types_reg_df = _merge_regressed(
    pitch_types,
    pitch_types_regressed,
    ["pitcher_mlbid", "name", "pitcher_hand", "season", "level_id", "pitch_tag", "game_type_group"],
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
