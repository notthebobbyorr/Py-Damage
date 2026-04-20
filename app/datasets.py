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
    """Aggregate grade_v13 to game_type_group level (weighted by n_pitches) and cast keys."""
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
    out["_wgrade"] = out["grade_v13"] * out["n_pitches"]
    grp_cols = [c for c in id_cols if c in out.columns]
    agg = (
        out.groupby(grp_cols)
        .agg(_wgrade_sum=("_wgrade", "sum"), _n=("n_pitches", "sum"))
        .reset_index()
    )
    agg["grade_v13"] = (agg["_wgrade_sum"] / agg["_n"]).round(1)
    out = agg.drop(columns=["_wgrade_sum", "_n"])
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
    pitcher_df = _pdf.merge(_exec_ps[_exec_ps_keys + ["grade_v13"]], on=_exec_ps_keys, how="left")
    del _pdf

if not _exec_pt.empty and not pitch_types.empty and "game_type_group" in pitch_types.columns:
    _ptdf = pitch_types.copy()
    for _c in ["pitcher_mlbid", "season", "level_id"]:
        if _c in _ptdf.columns:
            _ptdf[_c] = pd.to_numeric(_ptdf[_c], errors="coerce").astype("Int64")
    pitch_types = _ptdf.merge(_exec_pt[_exec_pt_keys + ["grade_v13"]], on=_exec_pt_keys, how="left")
    del _ptdf

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
            _p.groupby(_team_keys)
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
# Gamelog tables
# ---------------------------------------------------------------------------
hitter_gamelogs = load_csv("hitter_gamelogs.parquet")
pitcher_gamelogs = load_csv("pitcher_gamelogs.parquet")
pitch_type_gamelogs = load_csv("pitch_type_gamelogs.parquet")

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


hitter_gamelogs = _cast_gamelog_ints(hitter_gamelogs)
hitter_gamelogs = _normalize_gamelog_dates(hitter_gamelogs)
for _col in ["hitting_code", "opp_team"]:
    if _col in hitter_gamelogs.columns:
        hitter_gamelogs[_col] = _recode_team(hitter_gamelogs[_col], "AZ", "ARI")

pitcher_gamelogs = _cast_gamelog_ints(pitcher_gamelogs)
pitcher_gamelogs = _normalize_gamelog_dates(pitcher_gamelogs)
for _col in ["pitching_code", "opp_team"]:
    if _col in pitcher_gamelogs.columns:
        pitcher_gamelogs[_col] = _recode_team(pitcher_gamelogs[_col], "AZ", "ARI")

pitch_type_gamelogs = _cast_gamelog_ints(pitch_type_gamelogs)
pitch_type_gamelogs = _normalize_gamelog_dates(pitch_type_gamelogs)
for _col in ["pitching_code", "opp_team"]:
    if _col in pitch_type_gamelogs.columns:
        pitch_type_gamelogs[_col] = _recode_team(pitch_type_gamelogs[_col], "AZ", "ARI")

# Team gamelogs — aggregate counting stats per (team, game, date)
_T_H_KEYS = ["hitting_code", "game_pk", "game_date", "opp_team", "season", "level_id", "game_type_group"]
_T_H_SUM = ["PA", "pitches", "bbe", "damaged_bbe", "HR", "XBH", "hits",
            "la_lte_0_bbe", "la_gte_20_bbe", "swings", "chases", "whiffs", "BB", "K",
            "pulled_fbs", "selective_takes", "hittable_takes",
            "FA", "BR", "OFF", "vs_RHP", "vs_LHP"]

if not hitter_gamelogs.empty:
    _hgl_k = [c for c in _T_H_KEYS if c in hitter_gamelogs.columns]
    _hgl_s = [c for c in _T_H_SUM if c in hitter_gamelogs.columns]
    team_hitter_gamelogs = (
        hitter_gamelogs.groupby(_hgl_k, observed=True)[_hgl_s]
        .sum()
        .reset_index()
    )
    team_hitter_gamelogs = _cast_gamelog_ints(team_hitter_gamelogs)
    team_hitter_gamelogs = _normalize_gamelog_dates(team_hitter_gamelogs)
    del _hgl_k, _hgl_s
else:
    team_hitter_gamelogs = pd.DataFrame()

_T_P_KEYS = ["pitching_code", "game_pk", "game_date", "opp_team", "season", "level_id", "game_type_group"]
_T_P_SUM = ["TBF", "pitches", "bbe", "damaged_bbe", "HR", "XBH", "hits",
            "la_lte_0_bbe", "la_gte_20_bbe", "swings", "chases", "whiffs", "BB", "K",
            "zone_pitches", "FA", "BR", "OFF",
            "strikes", "balls", "out_of_zone", "vs_LHB", "vs_RHB"]

if not pitcher_gamelogs.empty:
    _pgl_k = [c for c in _T_P_KEYS if c in pitcher_gamelogs.columns]
    _pgl_s = [c for c in _T_P_SUM if c in pitcher_gamelogs.columns]
    team_pitcher_gamelogs = (
        pitcher_gamelogs.groupby(_pgl_k, observed=True)[_pgl_s]
        .sum()
        .reset_index()
    )
    team_pitcher_gamelogs = _cast_gamelog_ints(team_pitcher_gamelogs)
    team_pitcher_gamelogs = _normalize_gamelog_dates(team_pitcher_gamelogs)
    del _pgl_k, _pgl_s
else:
    team_pitcher_gamelogs = pd.DataFrame()

del _T_H_KEYS, _T_H_SUM, _T_P_KEYS, _T_P_SUM
