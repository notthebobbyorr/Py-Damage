from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.config import DATA_DIR
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
execution_pitcher = load_csv("location_v13_pitcher_season.parquet")
execution_pitch = load_csv("location_v13_pitcher_pitch.parquet")
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

# ---------------------------------------------------------------------------
# Season filter — drop pre-2020 rows from pitch-type and gamelog tables to
# reduce memory.  Hitter/pitcher seasonal aggregates retain full history.
# ---------------------------------------------------------------------------
_GAMELOG_MIN_SEASON = 2020


def _filter_min_season(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where season >= _GAMELOG_MIN_SEASON; pass through if no season col."""
    if df.empty or "season" not in df.columns:
        return df
    mask = pd.to_numeric(df["season"], errors="coerce") >= _GAMELOG_MIN_SEASON
    return df[mask].reset_index(drop=True)


pitch_types = _filter_min_season(pitch_types)
pitch_types_pct = _filter_min_season(pitch_types_pct)
pitch_types_regressed = _filter_min_season(pitch_types_regressed)
pitch_type_splits_df = _filter_min_season(pitch_type_splits_df)


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
# Merge takeoff_rate onto hitter_pct (for Takeoff% column on percentile page)
# ---------------------------------------------------------------------------
if not baserunning_reg.empty and not hitter_pct.empty and "game_type_group" in hitter_pct.columns:
    _br_hp = baserunning_reg.rename(columns={"runner_mlbid": "batter_mlbid"})
    _br_hp_want = ["batter_mlbid", "season", "level_id", "game_type_group", "takeoff_rate_raw"]
    _br_hp = (
        _br_hp[[c for c in _br_hp_want if c in _br_hp.columns]]
        .rename(columns={"takeoff_rate_raw": "takeoff_rate"})
        .drop_duplicates(subset=["batter_mlbid", "season", "level_id", "game_type_group"])
    )
    del _br_hp_want
    _hp = hitter_pct.copy()
    _hp_keys = ["batter_mlbid", "season", "level_id", "game_type_group"]
    for _c in ["batter_mlbid", "season", "level_id"]:
        _hp[_c] = pd.to_numeric(_hp[_c], errors="coerce").astype("Int64")
        _br_hp[_c] = pd.to_numeric(_br_hp[_c], errors="coerce").astype("Int64")
    hitter_pct = _hp.merge(_br_hp, on=_hp_keys, how="left")
    del _br_hp, _hp, _hp_keys, _c

# Merge takeoff_rate onto pitcher_pct (for Takeoff Against % on percentile page)
if not pitcher_baserunning_reg.empty and not pitcher_pct.empty and "game_type_group" in pitcher_pct.columns:
    _pbr_pp = pitcher_baserunning_reg.copy()
    _pbr_pp_want = ["pitcher_mlbid", "season", "level_id", "game_type_group", "takeoff_rate_raw"]
    _pbr_pp = (
        _pbr_pp[[c for c in _pbr_pp_want if c in _pbr_pp.columns]]
        .rename(columns={"takeoff_rate_raw": "takeoff_rate"})
        .drop_duplicates(subset=["pitcher_mlbid", "season", "level_id", "game_type_group"])
    )
    del _pbr_pp_want
    _pp = pitcher_pct.copy()
    _pp_keys = ["pitcher_mlbid", "season", "level_id", "game_type_group"]
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        _pp[_c] = pd.to_numeric(_pp[_c], errors="coerce").astype("Int64")
        _pbr_pp[_c] = pd.to_numeric(_pbr_pp[_c], errors="coerce").astype("Int64")
    pitcher_pct = _pp.merge(_pbr_pp, on=_pp_keys, how="left")
    del _pbr_pp, _pp, _pp_keys, _c

# Merge pitcher takeoff_rate onto pitch_types_pct (pitcher-level rate per pitch row)
if not pitcher_baserunning_reg.empty and not pitch_types_pct.empty and "game_type_group" in pitch_types_pct.columns:
    _pbr_ptpct = pitcher_baserunning_reg.copy()
    _pbr_ptpct_want = ["pitcher_mlbid", "season", "level_id", "game_type_group", "takeoff_rate_raw"]
    _pbr_ptpct = (
        _pbr_ptpct[[c for c in _pbr_ptpct_want if c in _pbr_ptpct.columns]]
        .rename(columns={"takeoff_rate_raw": "takeoff_rate"})
        .drop_duplicates(subset=["pitcher_mlbid", "season", "level_id", "game_type_group"])
    )
    del _pbr_ptpct_want
    _ptpct = pitch_types_pct.copy()
    _ptpct_keys = ["pitcher_mlbid", "season", "level_id", "game_type_group"]
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        _ptpct[_c] = pd.to_numeric(_ptpct[_c], errors="coerce").astype("Int64")
        _pbr_ptpct[_c] = pd.to_numeric(_pbr_ptpct[_c], errors="coerce").astype("Int64")
    pitch_types_pct = _ptpct.merge(_pbr_ptpct, on=_ptpct_keys, how="left")
    del _pbr_ptpct, _ptpct, _ptpct_keys, _c

# ---------------------------------------------------------------------------
# Merge execution grade (Location V13) onto pitcher_df and pitch_types
# ---------------------------------------------------------------------------
_GAME_TYPE_TO_GROUP = {
    "R": "Regular Season",
    "S": "Spring Training",
    "F": "Postseason", "D": "Postseason", "L": "Postseason", "W": "Postseason",
}

def _prep_execution(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Aggregate grade_v13 (and pred_v13_reg if present) to game_type_group level, weighted by n_pitches."""
    if df.empty or "grade_v13" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    if "level_id" in out.columns:
        out["level_id"] = out["level_id"].fillna(1)
    if "game_type" in out.columns:
        out["game_type"] = out["game_type"].fillna("R")
        out["game_type_group"] = out["game_type"].map(_GAME_TYPE_TO_GROUP).fillna("Regular Season")
    elif "game_type_group" not in out.columns:
        out["game_type_group"] = "Regular Season"
    has_reg = "pred_v13_reg" in out.columns
    out["_wgrade"] = out["grade_v13"] * out["n_pitches"]
    agg_kwargs: dict = {"_wgrade_sum": ("_wgrade", "sum"), "_n": ("n_pitches", "sum")}
    if has_reg:
        out["_wreg"] = out["pred_v13_reg"] * out["n_pitches"]
        agg_kwargs["_wreg_sum"] = ("_wreg", "sum")
    grp_cols = [c for c in id_cols if c in out.columns]
    agg = (
        out.groupby(grp_cols, observed=True)
        .agg(**agg_kwargs)
        .reset_index()
    )
    agg["grade_v13"] = (agg["_wgrade_sum"] / agg["_n"]).round(1)
    if has_reg:
        agg["pred_v13_reg"] = agg["_wreg_sum"] / agg["_n"]
    drop_cols = ["_wgrade_sum", "_n"] + (["_wreg_sum"] if has_reg else [])
    out = agg.drop(columns=drop_cols)
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in out.columns:
            out[_c] = pd.to_numeric(out[_c], errors="coerce").astype("Int64")
    return out

_exec_ps_keys = ["pitcher_mlbid", "season", "level_id", "game_type_group"]
_exec_pt_keys = ["pitcher_mlbid", "season", "level_id", "pitch_tag", "game_type_group"]

_exec_ps = _prep_execution(execution_pitcher, _exec_ps_keys)
_exec_pt = _prep_execution(execution_pitch, _exec_pt_keys)

if not _exec_ps.empty and not pitcher_df.empty and "game_type_group" in pitcher_df.columns:
    _pdf = pitcher_df.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _pdf.columns:
            _pdf[_c] = pd.to_numeric(_pdf[_c], errors="coerce").astype("Int64")
    _ps_val_cols = [c for c in ["grade_v13", "pred_v13_reg"] if c in _exec_ps.columns]
    pitcher_df = _pdf.merge(_exec_ps[_exec_ps_keys + _ps_val_cols], on=_exec_ps_keys, how="left")
    del _pdf, _ps_val_cols

if not _exec_pt.empty and not pitch_types.empty and "game_type_group" in pitch_types.columns:
    _ptdf = pitch_types.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _ptdf.columns:
            _ptdf[_c] = pd.to_numeric(_ptdf[_c], errors="coerce").astype("Int64")
    _pt_val_cols = [c for c in ["grade_v13", "pred_v13_reg"] if c in _exec_pt.columns]
    pitch_types = _ptdf.merge(_exec_pt[_exec_pt_keys + _pt_val_cols], on=_exec_pt_keys, how="left")
    del _ptdf, _pt_val_cols

# Merge grade_v13 onto pitcher_pct (Execution Grade on percentile page)
if not _exec_ps.empty and not pitcher_pct.empty and "game_type_group" in pitcher_pct.columns:
    _ppct = pitcher_pct.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _ppct.columns:
            _ppct[_c] = pd.to_numeric(_ppct[_c], errors="coerce").astype("Int64")
    pitcher_pct = _ppct.merge(_exec_ps[_exec_ps_keys + ["grade_v13"]], on=_exec_ps_keys, how="left")
    del _ppct

# Merge grade_v13 onto pitch_types_pct (Execution Grade on pitch percentile page)
if not _exec_pt.empty and not pitch_types_pct.empty and "game_type_group" in pitch_types_pct.columns:
    _ptpct2 = pitch_types_pct.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _ptpct2.columns:
            _ptpct2[_c] = pd.to_numeric(_ptpct2[_c], errors="coerce").astype("Int64")
    pitch_types_pct = _ptpct2.merge(_exec_pt[_exec_pt_keys + ["grade_v13"]], on=_exec_pt_keys, how="left")
    del _ptpct2

del _exec_ps, _exec_pt, _exec_ps_keys, _exec_pt_keys

# Merge predicted/damage metrics from pitcher_df into pitcher_pct (pct parquet may lack them)
_pred_cols = [c for c in ["p_SwStr_pct", "Damage_pct", "p_Damage_pct"] if c in pitcher_df.columns]
if _pred_cols and not pitcher_df.empty and not pitcher_pct.empty and "game_type_group" in pitcher_df.columns:
    _pred_keys = ["pitcher_mlbid", "season", "level_id", "game_type_group"]
    _psrc = pitcher_df.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _psrc.columns:
            _psrc[_c] = pd.to_numeric(_psrc[_c], errors="coerce").astype("Int64")
    _pdst = pitcher_pct.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _pdst.columns:
            _pdst[_c] = pd.to_numeric(_pdst[_c], errors="coerce").astype("Int64")
    for _col in _pred_cols:
        if _col in _pdst.columns:
            _pdst = _pdst.drop(columns=[_col])
    pitcher_pct = _pdst.merge(_psrc[_pred_keys + _pred_cols].drop_duplicates(_pred_keys), on=_pred_keys, how="left")
    del _psrc, _pdst

# Merge predicted/damage metrics from pitch_types into pitch_types_pct
_pred_cols_pt = [c for c in ["p_SwStr_pct", "Damage_pct", "p_Damage_pct"] if c in pitch_types.columns]
if _pred_cols_pt and not pitch_types.empty and not pitch_types_pct.empty and "game_type_group" in pitch_types.columns:
    _pred_keys_pt = ["pitcher_mlbid", "season", "level_id", "game_type_group", "pitch_tag"]
    _ptsrc = pitch_types.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _ptsrc.columns:
            _ptsrc[_c] = pd.to_numeric(_ptsrc[_c], errors="coerce").astype("Int64")
    _ptdst = pitch_types_pct.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _ptdst.columns:
            _ptdst[_c] = pd.to_numeric(_ptdst[_c], errors="coerce").astype("Int64")
    for _col in _pred_cols_pt:
        if _col in _ptdst.columns:
            _ptdst = _ptdst.drop(columns=[_col])
    pitch_types_pct = _ptdst.merge(_ptsrc[_pred_keys_pt + _pred_cols_pt].drop_duplicates(_pred_keys_pt), on=_pred_keys_pt, how="left")
    del _ptsrc, _ptdst
del _pred_cols, _pred_cols_pt

# ---------------------------------------------------------------------------
# Build team-level execution grade and merge onto team_stuff
# ---------------------------------------------------------------------------
if (
    "grade_v13" in pitcher_df.columns
    and not pitcher_df.empty
    and not team_stuff.empty
    and "game_type_group" in pitcher_df.columns
):
    _team_keys = ["pitching_code", "season", "level_id", "game_type_group"]
    _p = pitcher_df[
        [c for c in _team_keys + ["grade_v13", "TBF"] if c in pitcher_df.columns]
    ].dropna(subset=["grade_v13"])

    if not _p.empty and "TBF" in _p.columns:
        _p["_wgrade"] = _p["grade_v13"] * _p["TBF"]
        _team_exec = (
            _p.groupby(_team_keys, observed=True)
            .agg(_wgrade_sum=("_wgrade", "sum"), _n=("TBF", "sum"))
            .reset_index()
        )
        _team_exec["grade_v13"] = (_team_exec["_wgrade_sum"] / _team_exec["_n"]).round(1)
        _team_exec = _team_exec.drop(columns=["_wgrade_sum", "_n"])
        for _c in ["season", "level_id"]:
            _team_exec[_c] = pd.to_numeric(_team_exec[_c], errors="coerce").astype("Int64")
        _ts = team_stuff.copy()
        for _c in ["season", "level_id"]:
            if _c in _ts.columns:
                _ts[_c] = pd.to_numeric(_ts[_c], errors="coerce").astype("Int64")
        team_stuff = _ts.merge(_team_exec[_team_keys + ["grade_v13"]], on=_team_keys, how="left")
        del _ts
    del _p, _team_keys

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

# ---------------------------------------------------------------------------
# Gamelog helpers — no I/O at module level
# ---------------------------------------------------------------------------
_GAMELOG_INT_COLS = [
    "PA", "TBF", "pitches", "bbe", "damaged_bbe", "HR", "XBH", "hits",
    "la_lte_0_bbe", "la_gte_20_bbe", "swings", "chases", "whiffs", "BB", "K",
    "zone_pitches", "FA", "BR", "OFF",
    "strikes", "balls", "out_of_zone", "vs_LHB", "vs_RHB",
    "pulled_fbs", "selective_takes", "hittable_takes",
    "vs_RHP", "vs_LHP",
]


def _cast_gamelog_ints(df: pd.DataFrame) -> pd.DataFrame:
    for col in _GAMELOG_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _normalize_gamelog_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "game_date" not in df.columns:
        return df
    df = df.copy()
    _d = pd.to_datetime(df["game_date"], errors="coerce")
    df["game_date"] = _d.dt.date.astype(str).where(_d.notna(), "")
    return df


def _apply_stuff_grade(
    gamelog: pd.DataFrame,
    anchors: pd.DataFrame,
    key_cols: list[str],
    raw_col: str = "stuff_raw",
    grade_col: str = "stuff",
) -> pd.DataFrame:
    """
    Convert per-game raw stuff_raw to a 20-80 Pitch Grade using season-long
    p01/p99 anchors. anchors must have key_cols + ["stuff_p01", "stuff_p99"].
    Lower stuff_raw → higher grade (80 = best stuff).
    """
    if gamelog.empty or raw_col not in gamelog.columns or anchors.empty:
        return gamelog
    _keys = [c for c in key_cols if c in gamelog.columns and c in anchors.columns]
    if not _keys:
        return gamelog
    df = gamelog.merge(anchors[_keys + ["stuff_p01", "stuff_p99"]], on=_keys, how="left")
    valid = (
        df["stuff_p01"].notna() & df["stuff_p99"].notna()
        & (df["stuff_p99"] != df["stuff_p01"])
        & df[raw_col].notna()
    )
    span = df["stuff_p99"] - df["stuff_p01"]
    raw_g = 80.0 - 60.0 * (df[raw_col] - df["stuff_p01"]) / span
    df[grade_col] = pd.NA
    df.loc[valid, grade_col] = raw_g[valid].clip(20.0, 80.0).round(0)
    df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce").astype("Int64")
    return df.drop(columns=["stuff_p01", "stuff_p99"])


def _apply_exec_grade(
    gamelog: pd.DataFrame,
    anchors: pd.DataFrame,
    key_cols: list[str],
    raw_col: str = "pred_v13",
    grade_col: str = "grade_v13",
) -> pd.DataFrame:
    """
    Convert per-game mean pred_v13 to a 20-80 Execution Grade using
    season-long p1/p99 calibration anchors.
    Lower pred_v13 → higher grade (80 = best location).
    """
    if gamelog.empty or raw_col not in gamelog.columns or anchors.empty:
        return gamelog
    _keys = [c for c in key_cols if c in gamelog.columns and c in anchors.columns]
    if not _keys:
        return gamelog
    df = gamelog.merge(anchors[_keys + ["exec_p1", "exec_p99"]], on=_keys, how="left")
    valid = (
        df["exec_p1"].notna() & df["exec_p99"].notna()
        & (df["exec_p1"] != df["exec_p99"])
        & df[raw_col].notna()
    )
    span = df["exec_p99"] - df["exec_p1"]
    raw_g = 80.0 + (df[raw_col] - df["exec_p1"]) / span * (20.0 - 80.0)
    df[grade_col] = pd.NA
    df.loc[valid, grade_col] = raw_g[valid].clip(20.0, 80.0).round(0)
    df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce").astype("Int64")
    return df.drop(columns=["exec_p1", "exec_p99"])


# ---------------------------------------------------------------------------
# Precompute grade anchors from season-long distributions
# ---------------------------------------------------------------------------

# Stuff anchors for pitch-type gamelogs (by season + pitch_tag)
_anchor_stuff_pt = pd.DataFrame()
if not pitch_types_pct.empty and "stuff_raw" in pitch_types_pct.columns:
    _ptpct_mlb = pitch_types_pct.copy()
    if "level_id" in _ptpct_mlb.columns:
        _ptpct_mlb = _ptpct_mlb[pd.to_numeric(_ptpct_mlb["level_id"], errors="coerce") == 1]
    if "game_type_group" in _ptpct_mlb.columns:
        _ptpct_mlb = _ptpct_mlb[_ptpct_mlb["game_type_group"] == "Regular Season"]
    if "pitches" in _ptpct_mlb.columns:
        _ptpct_mlb = _ptpct_mlb[pd.to_numeric(_ptpct_mlb["pitches"], errors="coerce") >= 50]
    _ptpct_mlb = _ptpct_mlb.dropna(subset=["stuff_raw"])
    if not _ptpct_mlb.empty and "pitch_tag" in _ptpct_mlb.columns:
        _anchor_stuff_pt = (
            _ptpct_mlb.groupby(["season", "pitch_tag"], observed=True)["stuff_raw"]
            .agg(stuff_p01=lambda x: x.quantile(0.01), stuff_p99=lambda x: x.quantile(0.99))
            .reset_index()
        )
    del _ptpct_mlb

# Exec anchors for pitch-type gamelogs — use the same calibration file that
# location_v13_apply.py uses for season grades, so both use identical p1/p99 rubrics.
_anchor_exec_pt = pd.DataFrame()
_EXEC_CAL_PATH = DATA_DIR.parent.parent / "models" / "location_v13_grade_calibration.csv"
if _EXEC_CAL_PATH.exists():
    _cal = pd.read_csv(_EXEC_CAL_PATH)
    _cal_pooled = (
        _cal.groupby("pitch_tag")[["p1", "p99"]].median().reset_index()
        .rename(columns={"p1": "p1_pool", "p99": "p99_pool"})
    )
    if not execution_pitch.empty and "season" in execution_pitch.columns and "pitch_tag" in execution_pitch.columns:
        _seasons = execution_pitch["season"].dropna().unique()
        _rows = []
        for _s in _seasons:
            _s_int = int(_s)
            _cal_s = _cal[_cal["season"] == _s_int]
            for _pt in execution_pitch["pitch_tag"].dropna().unique():
                _match = _cal_s[_cal_s["pitch_tag"] == _pt]
                if not _match.empty:
                    _p1, _p99 = _match.iloc[0]["p1"], _match.iloc[0]["p99"]
                else:
                    _pool = _cal_pooled[_cal_pooled["pitch_tag"] == _pt]
                    _p1 = _pool.iloc[0]["p1_pool"] if not _pool.empty else None
                    _p99 = _pool.iloc[0]["p99_pool"] if not _pool.empty else None
                _rows.append({"season": _s_int, "pitch_tag": _pt, "exec_p1": _p1, "exec_p99": _p99})
        _anchor_exec_pt = pd.DataFrame(_rows).dropna(subset=["exec_p1", "exec_p99"])
    del _cal, _cal_pooled

# Convert season to int for join consistency
for _anc in [_anchor_stuff_pt, _anchor_exec_pt]:
    if not _anc.empty and "season" in _anc.columns:
        _anc["season"] = pd.to_numeric(_anc["season"], errors="coerce").astype("Int64")


def _merge_pitch_type_grades(
    pitcher_gl: pd.DataFrame,
    pitch_type_gl: pd.DataFrame,
    grade_cols: tuple[str, ...] = ("stuff", "grade_v13"),
    pitch_col: str = "pitches",
    join_keys: tuple[str, ...] = ("pitching_code", "game_pk"),
) -> pd.DataFrame:
    """
    Replace pitcher-game grade columns with a pitch-count weighted average of the
    per-pitch-type grades already computed in pitch_type_gl.  Only non-null grades
    contribute to the weighted average, so a pitch type with a missing grade is
    simply excluded rather than zeroed out.
    """
    if pitcher_gl.empty or pitch_type_gl.empty:
        return pitcher_gl
    _grade_cols = [c for c in grade_cols if c in pitch_type_gl.columns]
    _keys = [c for c in join_keys if c in pitch_type_gl.columns and c in pitcher_gl.columns]
    if not _grade_cols or not _keys or pitch_col not in pitch_type_gl.columns:
        return pitcher_gl

    ptgl = pitch_type_gl[_keys + [pitch_col] + _grade_cols].copy()
    ptgl[pitch_col] = pd.to_numeric(ptgl[pitch_col], errors="coerce")

    for gc in _grade_cols:
        gc_num = pd.to_numeric(ptgl[gc], errors="coerce")
        ptgl[f"_{gc}_w"] = gc_num * ptgl[pitch_col]
        ptgl[f"_{gc}_p"] = ptgl[pitch_col].where(gc_num.notna(), 0.0)

    agg_dict: dict = {}
    for gc in _grade_cols:
        agg_dict[f"_{gc}_wsum"] = (f"_{gc}_w", "sum")
        agg_dict[f"_{gc}_psum"] = (f"_{gc}_p", "sum")

    wgt = ptgl.groupby(_keys, observed=True).agg(**agg_dict).reset_index()

    for gc in _grade_cols:
        raw = wgt[f"_{gc}_wsum"] / wgt[f"_{gc}_psum"]
        wgt[gc] = pd.to_numeric(raw.round(0), errors="coerce").astype("Int64")

    wgt = wgt[_keys + _grade_cols]

    drop_cols = [c for c in _grade_cols if c in pitcher_gl.columns]
    if drop_cols:
        pitcher_gl = pitcher_gl.drop(columns=drop_cols)

    return pitcher_gl.merge(wgt, on=_keys, how="left")


# ---------------------------------------------------------------------------
# Lazy-loaded gamelog tables — deferred to first page access (~250 MB saved at startup)
# Rows are filtered to _GAMELOG_MIN_SEASON+ to reduce memory further.
# ---------------------------------------------------------------------------
_T_H_KEYS = ["hitting_code", "game_pk", "game_date", "opp_team", "season", "level_id", "game_type_group"]
_T_H_SUM = ["PA", "pitches", "bbe", "damaged_bbe", "HR", "XBH", "hits",
            "la_lte_0_bbe", "la_gte_20_bbe", "swings", "chases", "whiffs", "BB", "K",
            "pulled_fbs", "selective_takes", "hittable_takes",
            "FA", "BR", "OFF", "vs_RHP", "vs_LHP"]
_T_P_KEYS = ["pitching_code", "game_pk", "game_date", "opp_team", "season", "level_id", "game_type_group"]
_T_P_SUM = ["TBF", "pitches", "bbe", "damaged_bbe", "HR", "XBH", "hits",
            "la_lte_0_bbe", "la_gte_20_bbe", "swings", "chases", "whiffs", "BB", "K",
            "zone_pitches", "FA", "BR", "OFF",
            "strikes", "balls", "out_of_zone", "vs_LHB", "vs_RHB"]


@st.cache_resource
def get_hitter_gamelogs() -> pd.DataFrame:
    df = load_csv("hitter_gamelogs.parquet")
    if not df.empty and "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce") >= _GAMELOG_MIN_SEASON].copy()
    df = _cast_gamelog_ints(df)
    df = _normalize_gamelog_dates(df)
    for _col in ["hitting_code", "opp_team"]:
        if _col in df.columns:
            df[_col] = _recode_team(df[_col], "AZ", "ARI")
    return df


@st.cache_resource
def get_pitcher_gamelogs() -> pd.DataFrame:
    df = load_csv("pitcher_gamelogs.parquet")
    if not df.empty and "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce") >= _GAMELOG_MIN_SEASON].copy()
    df = _cast_gamelog_ints(df)
    df = _normalize_gamelog_dates(df)
    for _col in ["pitching_code", "opp_team"]:
        if _col in df.columns:
            df[_col] = _recode_team(df[_col], "AZ", "ARI")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df = _merge_pitch_type_grades(df, get_pitch_type_gamelogs())
    return df


@st.cache_resource
def get_pitch_type_gamelogs() -> pd.DataFrame:
    df = load_csv("pitch_type_gamelogs.parquet")
    if not df.empty and "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce") >= _GAMELOG_MIN_SEASON].copy()
    df = _cast_gamelog_ints(df)
    df = _normalize_gamelog_dates(df)
    for _col in ["pitching_code", "opp_team"]:
        if _col in df.columns:
            df[_col] = _recode_team(df[_col], "AZ", "ARI")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df = _apply_stuff_grade(df, _anchor_stuff_pt, ["season", "pitch_tag"])
    df = _apply_exec_grade(df, _anchor_exec_pt, ["season", "pitch_tag"])
    return df


@st.cache_resource
def get_team_hitter_gamelogs() -> pd.DataFrame:
    hgl = get_hitter_gamelogs()
    if hgl.empty:
        return pd.DataFrame()
    _hgl_k = [c for c in _T_H_KEYS if c in hgl.columns]
    _hgl_s = [c for c in _T_H_SUM if c in hgl.columns]
    result = hgl.groupby(_hgl_k, observed=True)[_hgl_s].sum().reset_index()
    result = _cast_gamelog_ints(result)
    result = _normalize_gamelog_dates(result)
    return result


@st.cache_resource
def get_team_pitcher_gamelogs() -> pd.DataFrame:
    pgl = get_pitcher_gamelogs()
    if pgl.empty:
        return pd.DataFrame()
    _pgl_k = [c for c in _T_P_KEYS if c in pgl.columns]
    _pgl_s = [c for c in _T_P_SUM if c in pgl.columns]
    result = pgl.groupby(_pgl_k, observed=True)[_pgl_s].sum().reset_index()
    result = _cast_gamelog_ints(result)
    result = _normalize_gamelog_dates(result)
    # TBF-weighted average of Pitch/Exec grades for each team game
    _grade_cols = [c for c in ["stuff", "grade_v13"] if c in pgl.columns]
    if _grade_cols and "TBF" in pgl.columns:
        _pg_g = pgl[
            [c for c in _pgl_k + ["TBF"] + _grade_cols if c in pgl.columns]
        ].copy()
        _pg_g["TBF"] = pd.to_numeric(_pg_g["TBF"], errors="coerce")
        for _gc in _grade_cols:
            _pg_g[f"_{_gc}_w"] = pd.to_numeric(_pg_g[_gc], errors="coerce") * _pg_g["TBF"]
        _wg_keys = [c for c in _pgl_k if c in _pg_g.columns]
        _wg_agg_dict = {f"_{_gc}_sum": (f"_{_gc}_w", "sum") for _gc in _grade_cols}
        _wg_agg_dict["_TBF"] = ("TBF", "sum")
        _wg_agg = _pg_g.groupby(_wg_keys, observed=True).agg(**_wg_agg_dict).reset_index()
        for _gc in _grade_cols:
            _wg_agg[_gc] = pd.to_numeric(
                (_wg_agg[f"_{_gc}_sum"] / _wg_agg["_TBF"]).round(0), errors="coerce"
            ).astype("Int64")
            _wg_agg = _wg_agg.drop(columns=[f"_{_gc}_sum"])
        _wg_agg = _wg_agg.drop(columns=["_TBF"])
        result = result.merge(
            _wg_agg[[c for c in _wg_keys + _grade_cols if c in _wg_agg.columns]],
            on=[c for c in _wg_keys if c in result.columns],
            how="left",
        )
    return result
