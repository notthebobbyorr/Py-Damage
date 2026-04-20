# -*- coding: utf-8 -*-
"""
Location V13 — Apply to New Data
Date: 2026-04-17

Score any pitch dataset with the trained Location V13 CatBoost model,
aggregate to pitcher-season and pitcher-pitch_tag levels, and produce
20-80 grades.

Key differences from V12 apply
-------------------------------
  - Uses per-pitcher personalized decision point (y_dp) derived from each
    pitcher's primary fastball velocity + extension, instead of the fixed
    26.417 ft constant used in V12.
  - pitch_velo and ext are loaded when available to compute y_dp; falls back
    to season-specific league averages (Y_DP_FALLBACK) when absent.
  - Grade column named pred_v13 / grade_v13.

Usage
-----
    python location_v13_apply.py --data-path /path/to/data.parquet [options]

    # Grade relative to historical calibration (recommended for single-season
    # or partial-season pulls):
    python location_v13_apply.py \\
        --data-path /path/to/data.parquet \\
        --cal-path  outputs/location_v13_grade_calibration.csv

    # Filter to one season:
    python location_v13_apply.py \\
        --data-path /path/to/data.parquet \\
        --seasons 2025

Required columns in input data
-------------------------------
    season, pitcher_mlbid, pitch_tag,
    x, z, throws, stands, batter_height,
    pi_zone_top, pi_zone_bottom,
    x0, vx0, ax, y0, vy0, ay, z0, vz0, az

Optional columns
----------------
    pitcher_name          — readable output
    description           — pitchout filtering (older schema)
    pitch_outcome         — pitchout filtering (newer schema)
    level_id              — defaults to 1 (MLB) if absent
    game_type             — defaults to "R" if absent
    pitch_velo, ext       — personalized y_dp; falls back to season averages
                            if absent or sparse

Rubric anchoring
----------------
    Grades are calibrated on level_id=1 (MLB) + game_type="R" (regular season).
    All levels and game_types are scored on the same scale; they appear as
    separate rows in the output.

Outputs (in --output-dir, default = same directory as input)
------------------------------------------------------------
    location_v13_scored.parquet      pitch-level, pred_v13 appended
    location_v13_pitcher_season.csv  pitcher x season [x level x game_type]
    location_v13_pitcher_pitch.csv   pitcher x season x pitch_tag [x level x game_type]
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor, Pool

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script file
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
MODEL_PATH  = _HERE.parent / "models" / "location_v13.cbm"
DEFAULT_CAL = _HERE.parent / "models" / "location_v13_grade_calibration.csv"

# ---------------------------------------------------------------------------
# Constants — must match training exactly
# ---------------------------------------------------------------------------
PLATE_HALF       = 0.7083
Y_PLATE          = 1.417
MIN_PITCHES_TYPE = 5    # min pitches for a pitcher-season-tag to enter arsenal lookup
MIN_N_PITCHER    = 20   # min pitches for a pitcher-season grade row
MIN_N_PITCH_TAG  = 5    # min pitches for a pitcher-pitch_tag grade row
MLB_LEVEL_ID     = 1
RUBRIC_GAME_TYPE = "R"  # rubric anchored to level_id=1 + game_type="R"

ARSENAL_TAGS = ["FA", "SI", "SL", "SW", "CU", "CH", "FS", "HC"]
TUNNEL_COLS  = [f"tunnel_dist_{t}" for t in ARSENAL_TAGS]
KIN_COLS     = ["x0", "vx0", "ax", "y0", "vy0", "ay", "z0", "vz0", "az"]

FEATURES_V13 = (
    ["x", "z", "x_norm", "z_norm", "euc_from_heart", "euc_from_zone", "batter_height"]
    + TUNNEL_COLS
    + ["pitch_tag", "throws", "stands"]
)
V13_CAT_IDX = [FEATURES_V13.index(c) for c in ["pitch_tag", "throws", "stands"]]

# Columns needed for scoring (minimum set)
REQUIRED_COLS = [
    "season", "pitcher_mlbid", "pitch_tag",
    "x", "z", "throws", "stands", "batter_height",
    "pi_zone_top", "pi_zone_bottom",
] + KIN_COLS

# Loaded when present; used for pitchout filtering, level/game grouping,
# personalized decision point, and display
OPTIONAL_COLS = [
    "pitcher_name",
    "description",
    "pitch_outcome",
    "level_id",
    "game_type",
    "game_pk",      # pass-through for game-level aggregation in gamelogs
    "pitch_velo",   # needed for personalized y_dp
    "ext",          # needed for personalized y_dp
]

# ---------------------------------------------------------------------------
# Personalized decision-point constants  (must match location_v13_train.py)
# ---------------------------------------------------------------------------
TAU_DP         = 0.175          # seconds — batter decision window before plate
PITCH_DIST     = 60.5           # ft — rubber to home plate
AY_REP         = 25.0           # ft/s² — representative aerodynamic deceleration
MPH_TO_FPS     = 5280.0 / 3600.0
FASTBALL_TAGS  = {"FA", "SI", "HC"}
MIN_FB_PITCHES = 10

# Season-specific league-average fallback y_dp (ft from plate).
# Future seasons clamp to the most recent known entry.
Y_DP_FALLBACK = {
    2021: 22.585,
    2022: 22.644,
    2023: 22.703,
    2024: 22.730,
    2025: 22.813,
}


def _fallback_y_dp(season: int) -> float:
    """Season-specific fallback y_dp, clamped to known range."""
    keys_le = [k for k in Y_DP_FALLBACK if k <= season]
    key = max(keys_le) if keys_le else min(Y_DP_FALLBACK)
    return Y_DP_FALLBACK[key]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------
def _normalize_name(name: str) -> str:
    """Strip accents and diacritics (e.g. Díaz → Diaz)."""
    return (
        unicodedata.normalize("NFD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_path: Path, seasons=None) -> pd.DataFrame:
    """Load parquet or CSV, filter pitchouts, validate required columns."""
    path = Path(data_path)
    print(f"Loading {path.name}...")

    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
        avail  = list(schema.keys())
        load   = [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in avail]
        miss   = [c for c in REQUIRED_COLS if c not in avail]
        if miss:
            print(f"  WARNING: missing required columns: {miss}", file=sys.stderr)
        df = pl.read_parquet(path, columns=load).to_pandas()
    else:
        df = pd.read_csv(path)

    if seasons:
        df = df[df["season"].isin(seasons)]

    # Pitchout filtering — two schema variants
    if "description" in df.columns:
        df = df[~df["description"].isin(["pitchout", "foul_pitchout"])]
    if "pitch_outcome" in df.columns:
        df = df[df["pitch_outcome"] != "NIP"]

    df = df[df["batter_height"].notna()]
    df = df.reset_index(drop=True)

    # Default level_id / game_type for historical MLB regular-season files
    if "level_id" not in df.columns:
        df["level_id"] = MLB_LEVEL_ID
    if "game_type" not in df.columns:
        df["game_type"] = RUBRIC_GAME_TYPE

    # Normalise pitcher names (strip diacritics)
    if "pitcher_name" in df.columns:
        df["pitcher_name"] = df["pitcher_name"].map(_normalize_name)

    level_counts = df.groupby(["level_id", "game_type"]).size().to_dict()
    print(
        f"  {len(df):,} pitches  |  "
        f"seasons: {sorted(df['season'].unique().tolist())}  |  "
        f"level×game_type: {level_counts}"
    )
    return df


# ---------------------------------------------------------------------------
# Location feature engineering
# ---------------------------------------------------------------------------
def build_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["x"] = df["x"].clip(-2.5, 2.5)
    df["z"] = df["z"].clip(0.0, 6.0)
    df["pi_zone_top"]    = df["pi_zone_top"].fillna(3.35)
    df["pi_zone_bottom"] = df["pi_zone_bottom"].fillna(1.70)

    top    = df["pi_zone_top"].values
    bot    = df["pi_zone_bottom"].values
    x      = df["x"].values
    z      = df["z"].values
    heart  = (top + bot) / 2
    zone_h = top - bot

    x_cl = np.clip(x, -PLATE_HALF, PLATE_HALF)
    z_cl = np.clip(z, bot, top)

    df["euc_from_heart"] = np.sqrt(x ** 2 + (z - heart) ** 2)
    df["euc_from_zone"]  = np.sqrt((x - x_cl) ** 2 + (z - z_cl) ** 2)
    df["x_norm"]         = x / PLATE_HALF
    df["z_norm"]         = np.where(zone_h > 0, (z - bot) / zone_h, 0.5)

    for col in ["pitch_tag", "throws", "stands"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# ---------------------------------------------------------------------------
# Tunnel physics
# ---------------------------------------------------------------------------
def _solve_quadratic_positive(a_c, b_c, c_c):
    disc = b_c ** 2 - 4 * a_c * c_c
    sd   = np.where(disc >= 0, np.sqrt(np.maximum(disc, 0)), np.nan)
    t1   = (-b_c + sd) / (2 * a_c)
    t2   = (-b_c - sd) / (2 * a_c)
    return np.where(
        (t1 > 0) & (t2 > 0), np.minimum(t1, t2),
        np.where(t1 > 0, t1, np.where(t2 > 0, t2, np.nan)),
    )


def _compute_tunnel_dist(
    sub: pd.DataFrame,
    comp_ax: np.ndarray,
    comp_ay: np.ndarray,
    comp_az: np.ndarray,
    y_decision: np.ndarray,
) -> np.ndarray:
    """
    Distance at the plate between each pitch's actual landing and where it
    would have landed if comp_{ax,ay,az} were applied after the decision point.

    y_decision is a per-row array (ft from plate).
    """
    x0  = sub["x0"].values;  y0  = sub["y0"].values;  z0  = sub["z0"].values
    vx0 = sub["vx0"].values; vy0 = sub["vy0"].values; vz0 = sub["vz0"].values
    ax  = sub["ax"].values;  ay  = sub["ay"].values;  az  = sub["az"].values

    t_d  = _solve_quadratic_positive(0.5 * ay, vy0, y0 - y_decision)
    x_d  = x0  + vx0 * t_d + 0.5 * ax  * t_d ** 2
    z_d  = z0  + vz0 * t_d + 0.5 * az  * t_d ** 2
    vx_d = vx0 + ax  * t_d
    vy_d = vy0 + ay  * t_d
    vz_d = vz0 + az  * t_d

    t2    = _solve_quadratic_positive(0.5 * comp_ay, vy_d, y_decision - Y_PLATE)
    x_tun = x_d + vx_d * t2 + 0.5 * comp_ax * t2 ** 2
    z_tun = z_d + vz_d * t2 + 0.5 * comp_az * t2 ** 2

    return np.sqrt((sub["x"].values - x_tun) ** 2 + (sub["z"].values - z_tun) ** 2)


# ---------------------------------------------------------------------------
# Per-pitcher decision point  (matches location_v13_train.py)
# ---------------------------------------------------------------------------
def build_pitcher_decision_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'y_dp' column: personalized decision-point y-coordinate (ft from plate).

    If pitch_velo / ext are absent from the dataframe, all rows receive the
    season-specific league-average fallback.

    Priority per pitcher-season:
      1. Mean pitch_velo + ext on FA / SI / HC (fastball family, >= MIN_FB_PITCHES)
      2. Mean pitch_velo + ext on most-thrown pitch (non-FB pitchers)
      3. Season-specific league average from Y_DP_FALLBACK
    """
    has_velo = "pitch_velo" in df.columns and df["pitch_velo"].notna().any()
    has_ext  = "ext"        in df.columns and df["ext"].notna().any()

    if not (has_velo and has_ext):
        # No velo/ext data — use season fallbacks for every row
        df["y_dp"] = df["season"].map(_fallback_y_dp)
        n_fallback = len(df)
        print(f"    pitch_velo/ext not available — all {n_fallback:,} pitches "
              f"use season-specific fallback y_dp")
        return df

    valid = df[df["pitch_velo"].notna() & df["ext"].notna()]

    # 1. Fastball-family params
    fb_params = (
        valid[valid["pitch_tag"].isin(FASTBALL_TAGS)]
        .groupby(["pitcher_mlbid", "season"])
        .agg(primary_velo=("pitch_velo", "mean"),
             primary_ext=("ext", "mean"),
             n_fb=("pitch_velo", "count"))
        .reset_index()
        .query(f"n_fb >= {MIN_FB_PITCHES}")
        [["pitcher_mlbid", "season", "primary_velo", "primary_ext"]]
    )

    # 2. Most-thrown-pitch fallback
    primary_tag = (
        valid.groupby(["pitcher_mlbid", "season", "pitch_tag"])
             .size().reset_index(name="n")
             .sort_values("n", ascending=False)
             .groupby(["pitcher_mlbid", "season"], as_index=False)
             .first()
             [["pitcher_mlbid", "season", "pitch_tag"]]
    )
    fallback_params = (
        valid.merge(primary_tag, on=["pitcher_mlbid", "season", "pitch_tag"])
             .groupby(["pitcher_mlbid", "season"])
             .agg(primary_velo=("pitch_velo", "mean"),
                  primary_ext=("ext", "mean"))
             .reset_index()
    )

    # 3. Combine: FB first, fill gaps with most-thrown
    params = fallback_params.merge(
        fb_params.rename(columns={"primary_velo": "fb_velo", "primary_ext": "fb_ext"}),
        on=["pitcher_mlbid", "season"], how="left",
    )
    params["primary_velo"] = params["fb_velo"].fillna(params["primary_velo"])
    params["primary_ext"]  = params["fb_ext"].fillna(params["primary_ext"])
    params = params[["pitcher_mlbid", "season", "primary_velo", "primary_ext"]]

    # 4. Compute y_dp from release speed + extension
    vy_rel  = -(params["primary_velo"].values * MPH_TO_FPS)
    y0_rel  = PITCH_DIST - params["primary_ext"].values
    t_plate = _solve_quadratic_positive(0.5 * AY_REP, vy_rel, y0_rel)
    t_dp    = t_plate - TAU_DP
    y_dp    = y0_rel + vy_rel * t_dp + 0.5 * AY_REP * t_dp ** 2
    params["y_dp"] = y_dp
    params = params.dropna(subset=["y_dp"])

    # 5. Merge onto dataset; fill remaining nulls with season fallback
    df = df.merge(
        params[["pitcher_mlbid", "season", "y_dp"]],
        on=["pitcher_mlbid", "season"], how="left",
    )
    null_mask  = df["y_dp"].isna()
    n_fallback = null_mask.sum()
    if n_fallback:
        df.loc[null_mask, "y_dp"] = df.loc[null_mask, "season"].map(_fallback_y_dp)

    n_pitchers = params["pitcher_mlbid"].nunique()
    print(f"    {n_pitchers:,} pitcher-seasons with personalized y_dp  |  "
          f"{n_fallback:,} pitches used season fallback")
    print(f"    y_dp range: [{df['y_dp'].min():.2f}, "
          f"{df['y_dp'].mean():.2f} mean, "
          f"{df['y_dp'].max():.2f}] ft")
    return df


# ---------------------------------------------------------------------------
# Tunnel features  (personalized y_dp)
# ---------------------------------------------------------------------------
def build_tunnel_features(
    df: pd.DataFrame, min_pitches_type: int = MIN_PITCHES_TYPE
) -> pd.DataFrame:
    """
    Compute tunnel_dist_{TAG} for every TAG in ARSENAL_TAGS using each
    pitcher's personalized decision point (y_dp).
    NaN when the pitcher doesn't throw TAG or the current pitch IS TAG.
    """
    print("  Building per-pitcher-season-tag kinematic lookup...")

    pt = (
        df.groupby(["pitcher_mlbid", "season", "pitch_tag"])
        .agg(
            comp_ax=("ax", "mean"),
            comp_ay=("ay", "mean"),
            comp_az=("az", "mean"),
            n=("ax", "count"),
        )
        .reset_index()
        .query(f"n >= {min_pitches_type}")
    )
    print(f"    {len(pt):,} pitcher-season-tag rows (>= {min_pitches_type} pitches)")

    for col in TUNNEL_COLS:
        df[col] = np.nan

    df = df.reset_index(drop=True)
    df["_rid"] = np.arange(len(df))
    MCOLS = ["_rid", "pitcher_mlbid", "season", "pitch_tag", "x", "z", "y_dp"] + KIN_COLS

    for comp_tag in ARSENAL_TAGS:
        cp = pt[pt["pitch_tag"] == comp_tag][
            ["pitcher_mlbid", "season", "comp_ax", "comp_ay", "comp_az"]
        ]
        if cp.empty:
            continue
        mask = df["pitch_tag"] != comp_tag
        sub  = df.loc[mask, MCOLS].merge(cp, on=["pitcher_mlbid", "season"], how="inner")
        if sub.empty:
            continue
        dists = _compute_tunnel_dist(
            sub,
            sub["comp_ax"].values, sub["comp_ay"].values, sub["comp_az"].values,
            y_decision=sub["y_dp"].values,
        )
        df.loc[sub["_rid"].values, f"tunnel_dist_{comp_tag}"] = dists

    df = df.drop(columns=["_rid"])

    print("  Per-type tunnel coverage:")
    for col in TUNNEL_COLS:
        print(f"    {col:<24} {df[col].notna().mean():.1%}")

    return df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    base_feats = ["x", "z", "x_norm", "z_norm", "euc_from_heart", "euc_from_zone",
                  "batter_height"]
    before = len(df)
    df = df.dropna(subset=base_feats).reset_index(drop=True)
    if before - len(df):
        print(f"  Dropped {before - len(df):,} rows for missing base features")

    df["pred_v13"] = model.predict(Pool(df[FEATURES_V13], cat_features=V13_CAT_IDX))
    print(
        f"  {len(df):,} pitches scored  "
        f"(mean={df['pred_v13'].mean():.4f}, "
        f"std={df['pred_v13'].std():.4f}, "
        f"range=[{df['pred_v13'].min():.4f}, {df['pred_v13'].max():.4f}])"
    )
    return df


# ---------------------------------------------------------------------------
# 20-80 grading helpers
# ---------------------------------------------------------------------------
def _grade_20_80(values: pd.Series, p1: pd.Series, p99: pd.Series) -> pd.Series:
    """Lower pred_v13 → higher grade (80 = best location)."""
    span  = (p99 - p1).fillna(0)
    valid = span > 0
    raw   = 80.0 + (values - p1) / span * (20.0 - 80.0)
    return pd.Series(
        np.where(valid, np.clip(raw, 20.0, 80.0), 50.0),
        index=values.index,
    )


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted percentile via linear interpolation. q in [0, 100]."""
    sort_idx = np.argsort(values)
    sv   = values[sort_idx]
    sw   = weights[sort_idx]
    cumw = np.cumsum(sw)
    cumw = cumw / cumw[-1]
    return float(np.interp(q / 100.0, cumw, sv))


def _hmean_weights(n_arr: np.ndarray) -> np.ndarray:
    """
    Harmonic-mean weights for calibration percentile computation.

    w_i = 2 * n_i * H / (n_i + H),  H = harmonic mean of the group.

    Downweights both high-usage specialists (survivorship bias) and
    very-low-usage fringe pitchers (noise).  Pitchers near the group
    harmonic mean receive maximum weight.
    """
    n = np.asarray(n_arr, dtype=float)
    H = len(n) / np.sum(1.0 / n)
    return 2.0 * n * H / (n + H)


def _hmean_p1_p99(sub: pd.DataFrame) -> pd.Series:
    """Harmonic-mean-weighted p1 / p99 for a calibration group."""
    vals    = sub["pred_v13"].values
    weights = _hmean_weights(sub["n_pitches"].values)
    return pd.Series({
        "p1":  _weighted_percentile(vals, weights, 1),
        "p99": _weighted_percentile(vals, weights, 99),
    })


def _resolve_calibration(pt: pd.DataFrame, cal_path) -> pd.DataFrame:
    """
    Merge p1/p99 calibration bounds into the pitcher-pitch_tag table.

    Priority:
      1. --cal-path file: per (pitch_tag, season) from training data.
         Falls back to pooled median per pitch_tag for seasons not in the file.
      2. Within-data: p1/p99 computed from level_id=1 + game_type="R" rows only.
    """
    if cal_path and Path(cal_path).exists():
        print(f"  Using calibration: {Path(cal_path).name}")
        cal = pd.read_csv(cal_path)

        pt = pt.merge(
            cal[["pitch_tag", "season", "p1", "p99"]],
            on=["pitch_tag", "season"],
            how="left",
        )

        unmatched = pt["p1"].isna()
        if unmatched.any():
            pooled = (
                cal.groupby("pitch_tag")[["p1", "p99"]]
                .median()
                .reset_index()
                .rename(columns={"p1": "p1_pool", "p99": "p99_pool"})
            )
            pt = pt.merge(pooled, on="pitch_tag", how="left")
            pt.loc[unmatched, "p1"]  = pt.loc[unmatched, "p1_pool"]
            pt.loc[unmatched, "p99"] = pt.loc[unmatched, "p99_pool"]
            n_fb = unmatched.sum()
            print(f"    {n_fb:,} pitcher-pitch_tag rows fell back to pooled calibration")
            pt = pt.drop(columns=["p1_pool", "p99_pool"])
    else:
        print(
            f"  Computing calibration within supplied data "
            f"(level_id={MLB_LEVEL_ID}, game_type='{RUBRIC_GAME_TYPE}' only)..."
        )
        mask = pd.Series(True, index=pt.index)
        if "level_id" in pt.columns:
            mask &= pt["level_id"] == MLB_LEVEL_ID
        if "game_type" in pt.columns:
            mask &= pt["game_type"] == RUBRIC_GAME_TYPE
        mlb_rows = pt[mask]
        ref = (
            mlb_rows.groupby(["pitch_tag", "season"])
            .apply(_hmean_p1_p99, include_groups=False)
            .reset_index()
        )
        pt = pt.merge(ref, on=["pitch_tag", "season"], how="left")

    return pt


def build_grades(
    df: pd.DataFrame,
    cal_path=None,
    min_n_pitcher: int = MIN_N_PITCHER,
    min_n_pitch_tag: int = MIN_N_PITCH_TAG,
):
    """
    Aggregate pred_v13 to pitcher-pitch_tag and pitcher-season levels,
    apply 20-80 grading. Returns (pitcher_season_df, pitcher_pitch_df).

    Group keys include level_id and game_type when present so that each
    level/game_type combination is a separate output row, graded on the
    MLB regular-season scale.
    """
    has_name      = "pitcher_name" in df.columns
    has_level     = "level_id"     in df.columns
    has_game_type = "game_type"    in df.columns

    id_cols_pt = []
    if has_name:      id_cols_pt.append("pitcher_name")
    id_cols_pt.append("pitcher_mlbid")
    id_cols_pt.append("season")
    if has_level:     id_cols_pt.append("level_id")
    if has_game_type: id_cols_pt.append("game_type")
    id_cols_pt.append("pitch_tag")

    id_cols_ps = [c for c in id_cols_pt if c != "pitch_tag"]

    # ── Pitcher × pitch_tag ───────────────────────────────────────────────
    pt = (
        df.groupby(id_cols_pt)
        .agg(pred_v13=("pred_v13", "mean"), n_pitches=("pred_v13", "count"))
        .reset_index()
        .query(f"n_pitches >= {min_n_pitch_tag}")
    )

    pt = _resolve_calibration(pt, cal_path)
    pt["grade_v13"] = _grade_20_80(pt["pred_v13"], pt["p1"], pt["p99"]).round(1)
    pt = pt.drop(columns=["p1", "p99"])

    # ── Pitcher × season (usage-weighted rollup) ──────────────────────────
    pt["_wgrade"] = pt["grade_v13"] * pt["n_pitches"]
    pt["_wpred"]  = pt["pred_v13"]  * pt["n_pitches"]

    ps = (
        pt.groupby(id_cols_ps)
        .agg(
            n_pitches=("n_pitches", "sum"),
            _wgrade=("_wgrade", "sum"),
            _wpred=("_wpred", "sum"),
        )
        .reset_index()
    )
    ps["grade_v13"] = (ps["_wgrade"] / ps["n_pitches"]).round(1)
    ps["pred_v13"]  =  ps["_wpred"]  / ps["n_pitches"]
    ps = ps.drop(columns=["_wgrade", "_wpred"])
    ps = ps.query(f"n_pitches >= {min_n_pitcher}")

    pt = pt.drop(columns=["_wgrade", "_wpred"])

    return ps, pt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Score new pitch data with the Location V13 model."
    )
    p.add_argument("--data-path", required=True,
                   help="Path to input parquet or CSV file")
    p.add_argument("--output-dir", default=None,
                   help="Directory for output files (default: same directory as --data-path)")
    p.add_argument("--seasons", nargs="+", type=int, default=None,
                   help="Restrict to these seasons, e.g. --seasons 2025")
    p.add_argument(
        "--cal-path", default=None,
        help=(
            "Pre-computed grade calibration CSV (columns: pitch_tag, season, p1, p99). "
            "If omitted and outputs/location_v13_grade_calibration.csv exists it is "
            "used automatically. Pass 'none' to force within-data calibration."
        ),
    )
    p.add_argument("--min-pitches", type=int, default=MIN_N_PITCHER,
                   help=f"Min pitcher-season pitches for a grade row (default {MIN_N_PITCHER})")
    p.add_argument("--min-pitches-type", type=int, default=MIN_PITCHES_TYPE,
                   help=(f"Min pitches per pitcher-season-tag for arsenal lookup "
                         f"(default {MIN_PITCHES_TYPE})"))
    p.add_argument("--model-path", default=str(MODEL_PATH),
                   help=f"Path to the .cbm model file (default: {MODEL_PATH})")
    p.add_argument("--chunk-by-season", action="store_true",
                   help="Save per-season parquet chunks (for pipeline stitching)")
    p.add_argument("--chunk-dir", default=None,
                   help="Directory for season-chunk parquets (requires --chunk-by-season)")
    return p.parse_args()


def main():
    args = parse_args()

    data_path  = Path(args.data_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_by_season = args.chunk_by_season
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else None
    if chunk_by_season and chunk_dir is None:
        raise ValueError("--chunk-dir is required when --chunk-by-season is set")
    if chunk_dir:
        chunk_dir.mkdir(parents=True, exist_ok=True)

    cal_path = None
    if args.cal_path and args.cal_path.lower() != "none":
        cal_path = Path(args.cal_path)
    elif DEFAULT_CAL.exists():
        cal_path = DEFAULT_CAL

    print("=" * 70)
    print("Location V13 — Apply")
    print("=" * 70)
    print(f"  model    : {model_path}")
    print(f"  data     : {data_path}")
    print(f"  output   : {output_dir}")
    print(f"  cal      : {cal_path or '(within-data)'}")

    # ── 1. Load ───────────────────────────────────────────────────────────
    df = load_data(data_path, args.seasons)

    # ── 2. Location features ──────────────────────────────────────────────
    print("\nBuilding location features...")
    df = build_location_features(df)

    # ── 3. Personalized decision points ───────────────────────────────────
    print("\nBuilding personalized decision points...")
    df = build_pitcher_decision_points(df)

    # ── 4. Tunnel features ────────────────────────────────────────────────
    print("\nBuilding per-type tunnel features...")
    df = build_tunnel_features(df, min_pitches_type=args.min_pitches_type)

    # ── 5. Score ──────────────────────────────────────────────────────────
    print(f"\nScoring with {model_path.name}...")
    df = score(df, model_path)

    # ── 6. Save pitch-level parquet ───────────────────────────────────────
    scored_path = output_dir / "location_v13_scored.parquet"
    drop_cols = [
        c for c in KIN_COLS + ["pi_zone_top", "pi_zone_bottom",
                                "euc_from_heart", "euc_from_zone",
                                "x_norm", "z_norm", "y_dp"]
        if c in df.columns
    ]
    df_out = df.drop(columns=drop_cols)
    df_out.to_parquet(scored_path, index=False)
    print(f"\n  Saved: {scored_path}")

    # ── 7. Grade ──────────────────────────────────────────────────────────
    print("\nBuilding 20-80 grades...")
    ps, pt = build_grades(df, cal_path=cal_path, min_n_pitcher=args.min_pitches)
    print(f"  {len(ps):,} pitcher-seasons  |  {len(pt):,} pitcher-pitch_tag rows")

    # Quick leaderboard — MLB regular season only
    id_col   = "pitcher_name" if "pitcher_name" in ps.columns else "pitcher_mlbid"
    top_cols = [c for c in [id_col, "season", "level_id", "game_type",
                             "n_pitches", "grade_v13", "pred_v13"]
                if c in ps.columns]
    ps_mlb = ps.copy()
    if "level_id"  in ps.columns: ps_mlb = ps_mlb[ps_mlb["level_id"]  == MLB_LEVEL_ID]
    if "game_type" in ps.columns: ps_mlb = ps_mlb[ps_mlb["game_type"] == RUBRIC_GAME_TYPE]

    print(f"\n--- Top 15 MLB regular season (best location) ---")
    print(ps_mlb.sort_values("grade_v13", ascending=False)
          .head(15)[top_cols]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n--- Bottom 15 MLB regular season (worst location) ---")
    print(ps_mlb.sort_values("grade_v13")
          .head(15)[top_cols]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ── 8. Save outputs ───────────────────────────────────────────────────
    sort_ps = ["season"]
    sort_pt = ["season"]
    for col in ["level_id", "game_type"]:
        if col in ps.columns: sort_ps.append(col)
        if col in pt.columns: sort_pt.append(col)
    sort_ps.append("grade_v13")
    sort_pt += ["pitch_tag", "grade_v13"]

    ps_sorted = ps.sort_values(sort_ps, ascending=[True] * (len(sort_ps) - 1) + [False])
    pt_sorted = pt.sort_values(sort_pt, ascending=[True] * (len(sort_pt) - 1) + [False])

    if chunk_by_season:
        # Save one parquet chunk per season so stitch_season_chunks can combine them
        for season_val, ps_chunk in ps_sorted.groupby("season"):
            out = chunk_dir / f"location_v13_pitcher_season.season_{season_val}.parquet"
            ps_chunk.to_parquet(out, index=False)
            print(f"  Saved chunk: {out.name}")
        for season_val, pt_chunk in pt_sorted.groupby("season"):
            out = chunk_dir / f"location_v13_pitcher_pitch.season_{season_val}.parquet"
            pt_chunk.to_parquet(out, index=False)
            print(f"  Saved chunk: {out.name}")
    else:
        ps_path = output_dir / "location_v13_pitcher_season.csv"
        pt_path = output_dir / "location_v13_pitcher_pitch.csv"
        ps_sorted.to_csv(ps_path, index=False)
        pt_sorted.to_csv(pt_path, index=False)
        print(f"\n  Saved: {ps_path}")
        print(f"  Saved: {pt_path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
