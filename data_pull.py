# -*- coding: utf-8 -*-
"""
Data Pull Script

Pulls pitch-level data from the database, applies feature engineering and model
predictions, and saves to a parquet file for aggregation.
"""
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import psycopg2

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency
    load_dotenv = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from catboost_predictor import CatBoostBaseballPredictor
except Exception:  # optional dependency
    CatBoostBaseballPredictor = None

try:
    from pygam import LinearGAM
except ImportError:  # optional dependency
    LinearGAM = None

try:
    from joblib import load as joblib_load
except ImportError:  # optional dependency
    joblib_load = None

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # optional dependency
    LabelEncoder = None

import pickle


DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR
MODEL_DIR = DATA_DIR

if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class DbConfig:
    dbname: str
    user: str
    password: str
    host: str
    port: int = 5432


@dataclass
class ModelBundle:
    whiff_loc: object | None = None
    whiff_base: object | None = None
    damage: object | None = None
    seager: object | None = None
    vaa_gam: object | None = None
    stuff_predictor: object | None = None


WHIFF_NUM_FEATURES = [
    "avg_release_z",
    "avg_release_x",
    "avg_ext",
    "pitch_velo",
    "rpm",
    "vbreak",
    "hbreak",
    "axis",
    "spin_efficiency",
    "z_angle_release",
    "x_angle_release",
    "vaa",
    "haa",
    "primary_velo",
    "primary_loc_adj_vaa",
    "primary_z_release",
    "primary_x_release",
    "primary_rpm",
    "primary_axis",
    "x",
    "z",
]
WHIFF_CAT_FEATURES = ["balls", "strikes", "throws", "stands"]
WHIFF_BASE_NUM_FEATURES = [
    feat for feat in WHIFF_NUM_FEATURES if feat not in {"x", "z"}
]
STUFF_SCALE_MEAN = 50.0
STUFF_SCALE_STD = 10.0


def _load_pickle(path: Path) -> object | None:
    if not path.exists():
        return None
    if joblib_load is not None:
        try:
            return joblib_load(path)
        except Exception:
            pass
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def load_models() -> ModelBundle:
    bundle = ModelBundle()
    whiff_loc_path = MODEL_DIR / "is_whiff_catboost_model_with_locations.cbm"
    whiff_base_path = MODEL_DIR / "is_whiff_catboost_model.cbm"
    damage_path = MODEL_DIR / "py_damage_model.pkl"
    seager_path = MODEL_DIR / "pySEAGER_model.pkl"
    vaa_gam_path = MODEL_DIR / "vaa_gam_model.pkl"
    stuff_path = MODEL_DIR / "catboost_baseball_model.cbm"

    if CatBoostClassifier is not None:
        if whiff_loc_path.exists():
            model = CatBoostClassifier()
            model.load_model(whiff_loc_path.as_posix())
            bundle.whiff_loc = model
            print(f"Loaded whiff model (with locations): {whiff_loc_path.name}")
        else:
            print(f"Whiff model with locations not found: {whiff_loc_path.name}")
        if whiff_base_path.exists():
            model = CatBoostClassifier()
            model.load_model(whiff_base_path.as_posix())
            bundle.whiff_base = model
            print(f"Loaded whiff model (base): {whiff_base_path.name}")
        else:
            print(f"Whiff model base not found: {whiff_base_path.name}")
    else:
        if whiff_loc_path.exists() or whiff_base_path.exists():
            print("CatBoost is not installed; skipping whiff models.")

    if LinearGAM is None and damage_path.exists():
        print("pyGAM is not installed; damage model may fail to load.")
    bundle.damage = _load_pickle(damage_path)
    print(
        f"Loaded damage model: {damage_path.name}"
        if bundle.damage is not None
        else f"Damage model failed or missing: {damage_path.name}"
    )
    if LinearGAM is None and seager_path.exists():
        print("pyGAM is not installed; SEAGER model may fail to load.")
    bundle.seager = _load_pickle(seager_path)
    print(
        f"Loaded SEAGER model: {seager_path.name}"
        if bundle.seager is not None
        else f"SEAGER model failed or missing: {seager_path.name}"
    )
    bundle.vaa_gam = _load_pickle(vaa_gam_path)
    print(
        f"Loaded VAA GAM model: {vaa_gam_path.name}"
        if bundle.vaa_gam is not None
        else f"VAA GAM model failed or missing: {vaa_gam_path.name}"
    )

    if stuff_path.exists():
        if CatBoostBaseballPredictor is not None:
            try:
                bundle.stuff_predictor = CatBoostBaseballPredictor(stuff_path)
            except Exception:
                print(f"Stuff model failed to load: {stuff_path.name}")
        elif CatBoostRegressor is None:
            print("CatBoost is not installed; skipping stuff model.")
    else:
        print(f"Stuff model not found: {stuff_path.name}")

    return bundle


def _ensure_columns(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return df
    return df.with_columns([pl.lit(None).alias(c) for c in missing])


def _predict_model(
    model: object, frame: pl.DataFrame, feature_cols: list[str]
) -> np.ndarray | None:
    if model is None:
        return None
    frame = _ensure_columns(frame, feature_cols)
    pdf = frame.select(feature_cols).to_pandas()
    try:
        if LinearGAM is not None and isinstance(model, LinearGAM):
            preds = model.predict(pdf.to_numpy())
        elif hasattr(model, "predict_proba"):
            preds = model.predict_proba(pdf)[:, 1]
        else:
            preds = model.predict(pdf)
        return np.asarray(preds)
    except Exception:
        return None


def _get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def load_db_config() -> DbConfig:
    return DbConfig(
        dbname=_get_env("POOBAH_DB", "cage"),
        user=_get_env("POOBAH_USER"),
        password=_get_env("POOBAH_PASSWORD"),
        host=_get_env("POOBAH_HOST", "scully.baseballprospectus.com"),
        port=int(_get_env("POOBAH_PORT", "5432")),
    )


def read_pitch_data(
    cfg: DbConfig, min_season: int, max_season: int, level_ids: Iterable[int]
) -> pl.DataFrame:
    level_clause = ",".join(str(int(x)) for x in level_ids)
    query = f"""
        SELECT
            a.season,
            a.game_pk,
            a.game_date,
            a.game_type,
            a.level_id,
            a.inning,
            a.half_inning,
            a.home_mlbid,
            a.away_mlbid,
            a.at_bat_index,
            a.event_index,
            a.pitch_of_ab,
            a.swing_type,
            a.is_contact,
            a.is_in_play,
            a.is_strike,
            a.is_ball,
            a.strikes_before,
            a.balls_before,
            a.batter_mlbid,
            a.pitcher_mlbid,
            a.batter_hand,
            a.pitcher_hand,
            a.batter_name_first,
            a.batter_name_last,
            a.pitcher_name_first,
            a.pitcher_name_last,
            a.pitcher_role,
            a.batter_position,
            a.pitch_outcome,
            a.is_inzone_pi,
            a.pi_zone_top,
            a.pi_zone_bottom,
            a.pi_pitch_group,
            a.pi_pitch_type,
            a.pi_pitch_sub_type,
            coalesce(a.px_corr, a.px_adj, a.px_orig) as x,
            coalesce(a.pz_corr, a.pz_adj, a.pz_orig) as z,
            coalesce(a.x_angle_plate_adj, a.x_angle_plate_corr) as haa,
            coalesce(a.z_angle_plate_adj, a.z_angle_plate_corr) as vaa,
            coalesce(a.start_speed_55_corr, a.start_speed_55_adj, a.start_speed_orig) as pitch_velo,
            coalesce(a.end_speed_orig, a.end_speed_adj, a.end_speed_corr) as end_velo,
            coalesce(a.plate_time_orig) as plate_time,
            coalesce(a.x_angle_release_adj, a.x_angle_release_corr) as x_angle_release,
            coalesce(a.z_angle_release_adj, a.z_angle_release_corr) as z_angle_release,
            coalesce(a.extension_orig, 60.5 - a.y_release_adj, 60.5 - a.y_release_corr) as ext,
            coalesce(a.obs_spin_rate_orig, a.obs_spin_rate_corr) as rpm,
            coalesce(a.inf_spin_rate_corr, a.inf_spin_rate_adj, a.inf_spin_rate_orig) as inf_rpm,
            coalesce(a.pfx_x_corr, a.pfx_x_adj) as pfx_x_short,
            coalesce(a.pfx_z_corr, a.pfx_z_adj) as pfx_z_short,
            coalesce(a.x55_corr, a.x55_adj) as x55,
            coalesce(a.z55_corr, a.z55_adj) as z55,
            coalesce(a.x_release_corr, a.x_release_adj) as release_x,
            coalesce(a.z_release_corr, a.z_release_adj) as release_z,
            coalesce(a.obs_spin_axis_orig, a.obs_spin_axis_corr) as axis,
            coalesce(a.inf_spin_axis_corr, a.inf_spin_axis_adj, a.inf_spin_axis_orig) as inf_axis,
            a.park_mlbid,
            a.elevation,
            a.temperature_game,
            a.batter_height,
            a.pitcher_height,
            a.vx0_orig as vx0,
            a.vy0_orig as vy0,
            a.vz0_orig as vz0,
            a.x0_orig as x0,
            a.y0_orig as y0,
            a.z0_orig as z0,
            coalesce(a.ax_adj, a.ax_corr) as ax,
            coalesce(a.ay_adj, a.ay_corr) as ay,
            coalesce(a.az_adj, a.az_corr) as az,
            sp.launch_speed as exit_velo,
            sp.launch_angle,
            sp.coord_x as hc_x,
            sp.coord_y as hc_y,
            sc.home_team,
            sc.away_team,
            sc.events,
            sc.description,
            sc.des,
            sc.bb_type,
            sc.arm_angle,
            sc.attack_angle,
            sc.attack_direction,
            sc.swing_path_tilt,
            sc.intercept_ball_minus_batter_pos_x_inches,
            sc.intercept_ball_minus_batter_pos_y_inches,
            sc.bat_speed,
            sc.swing_length,
            sc.estimated_woba_using_speedangle as xwoba,
            sc.estimated_ba_using_speedangle as xba,
            sc.estimated_slg_using_speedangle as xlsg,
            sc.api_break_z_with_gravity as ivb,
            sc.api_break_x_arm as hb_arm,
            sc.api_break_x_batter_in as hb_batter_in,
            sc.woba_value,
            sc.delta_run_exp,
            sc.n_thruorder_pitcher
        FROM pitchinfo.pitches_public a
        LEFT JOIN mlbapi.batted_balls sp
            ON a.game_pk = sp.game_pk
            AND a.pitch_of_ab = sp.pitch_number
            AND a.at_bat_index = sp.at_bat_index
            AND a.event_index = sp.event_index
        LEFT JOIN savant.savant_pbp sc
            ON a.game_pk = sc.game_pk
            AND a.pitch_of_ab = sc.pitch_number
            AND (a.at_bat_index + 1) = sc.at_bat_number
        WHERE a.season >= {min_season} AND a.season <= {max_season}
            AND a.level_id IN ({level_clause})
            AND a.game_type = 'R'
    """
    with psycopg2.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
    ) as conn:
        print(
            f"Running pitch query for seasons {min_season}-{max_season} and levels {level_clause}..."
        )
        df = pd.read_sql_query(query, conn)
        print(f"Fetched {len(df):,} pitch rows.")
    return pl.from_pandas(df)


def add_baseout(cfg: DbConfig, pitch_df: pl.DataFrame) -> pl.DataFrame:
    if pitch_df.is_empty():
        return pitch_df
    game_pks = pitch_df.get_column("game_pk").unique().to_list()
    if not game_pks:
        return pitch_df
    ids = ",".join(str(int(x)) for x in game_pks if x is not None)
    print(f"Fetching baseout rows for {len(game_pks):,} games...")
    query = f"SELECT * FROM mlbapi.baseout WHERE game_pk IN ({ids})"
    with psycopg2.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
    ) as conn:
        baseout = pd.read_sql_query(query, conn)
    print(f"Fetched {len(baseout):,} baseout rows.")
    baseout_pl = pl.from_pandas(baseout)
    return pitch_df.join(
        baseout_pl,
        on=["game_pk", "at_bat_index", "event_index"],
        how="left",
    )


def _tag_pitch(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("pi_pitch_sub_type") == "SW")
        .then(pl.lit("SW"))
        .when(pl.col("pi_pitch_sub_type") == "SP")
        .then(pl.lit("SP"))
        .when(pl.col("pi_pitch_type") == "SI")
        .then(pl.lit("SI"))
        .when((pl.col("pi_pitch_group") == "FA") & (pl.col("pi_pitch_type") == "FC"))
        .then(pl.lit("HC"))
        .when(pl.col("pi_pitch_type") == "FS")
        .then(pl.lit("FS"))
        .when(pl.col("pi_pitch_type") == "FA")
        .then(pl.lit("FA"))
        .when(pl.col("pi_pitch_group") == "SL")
        .then(pl.lit("SL"))
        .when(pl.col("pi_pitch_type") == "CH")
        .then(pl.lit("CH"))
        .when(pl.col("pi_pitch_group") == "CU")
        .then(pl.lit("CU"))
        .otherwise(pl.lit("XX"))
        .alias("pitch_tag")
    )


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = df.with_columns(
        [
            (pl.col("hc_x") - 125.42).alias("hc_x_adj"),
            (198.27 - pl.col("hc_y")).alias("hc_y_adj"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("batter_hand") == "L")
            .then(
                -1
                * (
                    pl.arctan2(pl.col("hc_x_adj"), pl.col("hc_y_adj"))
                    * (180 / np.pi)
                    * 0.75
                )
            )
            .otherwise(
                pl.arctan2(pl.col("hc_x_adj"), pl.col("hc_y_adj"))
                * (180 / np.pi)
                * 0.75
            )
            .alias("spray_angle_adj"),
            (
                pl.col("batter_name_first") + pl.lit(" ") + pl.col("batter_name_last")
            ).alias("hitter_name"),
            (
                pl.col("pitcher_name_first") + pl.lit(" ") + pl.col("pitcher_name_last")
            ).alias("pitcher_name"),
            (
                pl.col("pitcher_name_first") + pl.lit(" ") + pl.col("pitcher_name_last")
            ).alias("name"),
            pl.when(pl.col("batter_hand") == "L")
            .then(-1 * pl.col("x"))
            .otherwise(pl.col("x"))
            .alias("x_adj"),
            pl.when(pl.col("batter_hand") == "L")
            .then(-1 * pl.col("release_x"))
            .otherwise(pl.col("release_x"))
            .alias("release_x_adj"),
            pl.when(pl.col("batter_hand") == "L")
            .then(-1 * pl.col("haa"))
            .otherwise(pl.col("haa"))
            .alias("haa_adj"),
            pl.when(pl.col("pi_pitch_group") == "FA")
            .then(pl.lit("FA"))
            .when(pl.col("pi_pitch_group").is_in(["SL", "CU"]))
            .then(pl.lit("BR"))
            .when(pl.col("pi_pitch_group") == "CH")
            .then(pl.lit("OFF"))
            .otherwise(pl.lit("XX"))
            .alias("pitch_group"),
            pl.when(pl.col("batter_hand") == "L")
            .then(-1 * pl.col("pfx_x_short"))
            .otherwise(pl.col("pfx_x_short"))
            .alias("pfx_x_short_adj"),
            pl.when(pl.col("half_inning") == "bottom")
            .then(pl.coalesce("home_team", pl.col("home_mlbid").cast(pl.Utf8)))
            .otherwise(pl.coalesce("away_team", pl.col("away_mlbid").cast(pl.Utf8)))
            .alias("hitting_code"),
            pl.when(pl.col("half_inning") == "bottom")
            .then(pl.coalesce("away_team", pl.col("away_mlbid").cast(pl.Utf8)))
            .otherwise(pl.coalesce("home_team", pl.col("home_mlbid").cast(pl.Utf8)))
            .alias("pitching_code"),
            pl.col("balls_before").alias("balls"),
            pl.col("strikes_before").alias("strikes"),
            pl.col("pitcher_hand").alias("throws"),
            pl.col("batter_hand").alias("stands"),
            (pl.col("outs_end") - pl.col("outs_start")).alias("outs_recorded"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("swing_type") == "swing")
            .then(1)
            .otherwise(0)
            .alias("swing"),
            pl.when(pl.col("swing_type") == "swing")
            .then(1)
            .otherwise(0)
            .alias("is_swing"),
            pl.when((pl.col("is_contact") == False) & (pl.col("swing_type") == "swing"))
            .then(1)
            .otherwise(0)
            .alias("whiff"),
            pl.when(pl.col("is_in_play") == True).then(1).otherwise(0).alias("bbe"),
            pl.when((pl.col("pitch_outcome") == "B") & (pl.col("balls_before") == 3))
            .then(1)
            .otherwise(0)
            .alias("bb"),
            pl.when((pl.col("pitch_outcome") == "S") & (pl.col("strikes_before") == 2))
            .then(1)
            .otherwise(0)
            .alias("k"),
            (1.8 * pl.col("pfx_z_short")).alias("vbreak"),
            (1.8 * pl.col("pfx_x_short")).alias("hbreak"),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("game_pk").cast(str) + "_" + pl.col("at_bat_index").cast(str)
            ).alias("pa_id"),
            (pl.col("balls").cast(str) + "-" + pl.col("strikes").cast(str)).alias(
                "count"
            ),
        ]
    )

    spin_eff = (pl.col("inf_rpm") / pl.col("rpm")).clip(0, 1)

    df = df.with_columns(
        [
            pl.lit(None).cast(pl.Float64).alias("xwt"),
            pl.lit(None).cast(pl.Float64).alias("xgb_woba"),
            pl.lit(None).cast(pl.Float64).alias("damage_pred"),
            pl.lit(None).cast(pl.Float64).alias("decision_value"),
            spin_eff.alias("spin_efficiency"),
            pl.lit(None).cast(pl.Float64).alias("pred_whiff_loc"),
            pl.lit(None).cast(pl.Float64).alias("pred_whiff_base"),
            pl.lit(None).cast(pl.Float64).alias("stuff_raw"),
        ]
    )
    return df


def apply_vaa_gam(df: pl.DataFrame, model: object | None) -> pl.DataFrame:
    if df.is_empty():
        return df
    if model is None:
        return df.with_columns(pl.col("vaa").alias("location_vaa"))
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = ["z", "pitch_velo"]
    preds = _predict_model(model, df, feature_cols)
    if preds is None:
        return df.with_columns(pl.col("vaa").alias("location_vaa"))
    return df.with_columns(
        (pl.col("vaa") - pl.Series("location_vaa", preds)).alias("loc_adj_vaa")
    )


def add_pitcher_context(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    averages = df.group_by(["pitcher_mlbid", "level_id", "season"]).agg(
        [
            pl.mean("release_z").alias("avg_release_z"),
            pl.mean("release_x").alias("avg_release_x"),
            pl.mean("ext").alias("avg_ext"),
            pl.mean("arm_angle").alias("arm_angle"),
        ]
    )
    return df.join(averages, on=["pitcher_mlbid", "level_id", "season"], how="left")


def add_primary_pitch_context(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _tag_pitch(df)
    counts = (
        df.filter(pl.col("pitch_tag").is_in(["FA", "SI", "HC", "SP"]))
        .group_by(["pitcher_mlbid", "level_id", "season", "stands", "pitch_tag"])
        .agg(pl.len().alias("pitch_count"))
        .sort(
            [
                "pitcher_mlbid",
                "level_id",
                "season",
                "stands",
                "pitch_tag",
                "pitch_count",
            ],
            descending=[False, False, False, False, False, True],
        )
    )
    primary = counts.group_by(["pitcher_mlbid", "level_id", "season", "stands"]).agg(
        pl.first("pitch_tag").alias("primary_tag")
    )
    df = df.join(
        primary, on=["pitcher_mlbid", "level_id", "season", "stands"], how="left"
    )
    primary_stats = (
        df.filter(pl.col("pitch_tag") == pl.col("primary_tag"))
        .group_by(["pitcher_mlbid", "level_id", "season", "stands", "primary_tag"])
        .agg(
            [
                pl.mean("pitch_velo").alias("primary_velo"),
                pl.mean("vbreak").alias("primary_vbreak"),
                pl.mean("hbreak").alias("primary_hbreak"),
                pl.mean("vaa").alias("primary_vaa"),
                pl.mean("loc_adj_vaa").alias("primary_loc_adj_vaa"),
                pl.mean("z_angle_release").alias("primary_z_release"),
                pl.mean("x_angle_release").alias("primary_x_release"),
                pl.mean("rpm").alias("primary_rpm"),
                pl.mean("axis").alias("primary_axis"),
            ]
        )
    )
    return df.join(
        primary_stats,
        on=["pitcher_mlbid", "level_id", "season", "stands", "primary_tag"],
        how="left",
    )


def predict_catboost_probs(
    model: object,
    df: pl.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> np.ndarray | None:
    if model is None:
        return None
    df = _ensure_columns(df, feature_cols)
    pdf = df.select(feature_cols).to_pandas()
    for col in cat_cols:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype(str)
    try:
        return model.predict_proba(pdf)[:, 1]
    except Exception:
        return None


def apply_models(df: pl.DataFrame, models: ModelBundle) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = apply_vaa_gam(df, models.vaa_gam)
    df = add_pitcher_context(df)
    df = add_primary_pitch_context(df)

    loc_preds = predict_catboost_probs(
        models.whiff_loc,
        df,
        WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES,
        WHIFF_CAT_FEATURES,
    )
    if loc_preds is not None:
        df = df.with_columns(pl.Series(name="pred_whiff_loc", values=loc_preds))
        print(f"Applied whiff model (with locations): {len(loc_preds):,} predictions")
    else:
        print("Skipped whiff model (with locations).")

    base_preds = predict_catboost_probs(
        models.whiff_base,
        df,
        WHIFF_BASE_NUM_FEATURES + WHIFF_CAT_FEATURES,
        WHIFF_CAT_FEATURES,
    )
    if base_preds is not None:
        df = df.with_columns(pl.Series(name="pred_whiff_base", values=base_preds))
        print(f"Applied whiff model (base): {len(base_preds):,} predictions")
    else:
        print("Skipped whiff model (base).")

    if models.vaa_gam is not None:
        if LinearGAM is not None and isinstance(models.vaa_gam, LinearGAM):
            vaa_features = ["z", "pitch_velo"]
            print(f"VAA model features (LinearGAM): {vaa_features}")
        else:
            if hasattr(models.vaa_gam, "feature_names_in_"):
                vaa_features = list(models.vaa_gam.feature_names_in_)
            else:
                vaa_features = ["z", "pitch_velo"]
            print(f"VAA model features: {vaa_features}")

    if models.damage is not None:
        if LinearGAM is not None and isinstance(models.damage, LinearGAM):
            damage_features = ["spray_angle_adj_full", "launch_angle_full"]
            print(f"Damage model features (LinearGAM): {damage_features}")
            df = df.with_columns(
                [
                    pl.col("spray_angle_adj")
                    .replace([np.inf, -np.inf], None)
                    .fill_null(0)
                    .alias("spray_angle_adj_full"),
                    pl.col("launch_angle")
                    .replace([np.inf, -np.inf], None)
                    .fill_null(0)
                    .alias("launch_angle_full"),
                ]
            )
            damage_preds = _predict_model(models.damage, df, damage_features)
            if damage_preds is not None:
                df = df.with_columns(pl.Series(name="damage_pred", values=damage_preds))
                print(f"Applied damage model: {len(damage_preds):,} predictions")
            else:
                print("Damage model loaded but predictions failed.")
        else:
            if hasattr(models.damage, "feature_names_in_"):
                damage_features = list(models.damage.feature_names_in_)
            else:
                damage_features = ["exit_velo", "launch_angle", "spray_angle_adj"]
            print(f"Damage model features: {damage_features}")
            damage_frame = df.filter(pl.col("is_in_play") == True)
            damage_preds = _predict_model(models.damage, damage_frame, damage_features)
            if damage_preds is not None:
                damage_pred_series = pl.Series(name="damage_pred", values=damage_preds)
                df = df.with_columns(
                    pl.when(pl.col("is_in_play") == True)
                    .then(damage_pred_series)
                    .otherwise(pl.lit(None))
                    .alias("damage_pred")
                )
                print(f"Applied damage model: {len(damage_preds):,} predictions")
            else:
                print("Damage model loaded but predictions failed.")
    else:
        print("Skipped damage model.")

    if models.seager is not None:
        if LinearGAM is not None and isinstance(models.seager, LinearGAM):
            seager_features = ["x", "z", "balls", "strikes", "stands"]
            print(f"SEAGER model features (LinearGAM): {seager_features}")
        else:
            if hasattr(models.seager, "feature_names_in_"):
                seager_features = list(models.seager.feature_names_in_)
            else:
                seager_features = [
                    "balls",
                    "strikes",
                    "x",
                    "z",
                    "pitch_velo",
                    "vaa",
                    "haa",
                    "stands",
                    "throws",
                ]
            print(f"SEAGER model features: {seager_features}")

        seager_df = df.select(seager_features).to_pandas()

        if LabelEncoder is not None:
            if "stands" in seager_features:
                le_stands = LabelEncoder()
                seager_df["stands"] = seager_df["stands"].fillna("R")
                seager_df["stands"] = le_stands.fit_transform(seager_df["stands"])
            if "throws" in seager_features:
                le_throws = LabelEncoder()
                seager_df["throws"] = seager_df["throws"].fillna("R")
                seager_df["throws"] = le_throws.fit_transform(seager_df["throws"])
        else:
            if "stands" in seager_features:
                seager_df["stands"] = (
                    seager_df["stands"].map({"L": 0, "R": 1}).fillna(1)
                )
            if "throws" in seager_features:
                seager_df["throws"] = (
                    seager_df["throws"].map({"L": 0, "R": 1}).fillna(1)
                )

        if "balls" in seager_features:
            seager_df["balls"] = seager_df["balls"].astype("Int64")
        if "strikes" in seager_features:
            seager_df["strikes"] = seager_df["strikes"].astype("Int64")

        try:
            if LinearGAM is not None and isinstance(models.seager, LinearGAM):
                seager_preds = models.seager.predict(
                    seager_df[seager_features].to_numpy()
                )
            elif hasattr(models.seager, "predict_proba"):
                seager_preds = models.seager.predict_proba(seager_df)[:, 1]
            else:
                seager_preds = models.seager.predict(seager_df)
            seager_preds = np.asarray(seager_preds)
        except Exception as e:
            print(f"SEAGER model prediction failed: {e}")
            seager_preds = None

        if seager_preds is not None:
            df = df.with_columns(
                pl.Series(name="decision_value_raw", values=seager_preds)
            )
            df = df.with_columns(
                pl.when(pl.col("swing") == 0)
                .then(-1 * pl.col("decision_value_raw"))
                .otherwise(pl.col("decision_value_raw"))
                .alias("decision_value")
            )
            print(f"Applied SEAGER model: {len(seager_preds):,} predictions")
        else:
            print("SEAGER model loaded but predictions failed.")
    else:
        print("Skipped SEAGER model.")

    stuff_preds = predict_stuff(models.stuff_predictor, df)
    if stuff_preds is not None:
        df = df.with_columns(pl.Series(name="stuff_raw", values=stuff_preds))
        print(f"Applied stuff model: {len(stuff_preds):,} predictions")
    else:
        print("Skipped stuff model.")

    return df


def predict_stuff(predictor: object | None, df: pl.DataFrame) -> np.ndarray | None:
    if predictor is None:
        return None
    try:
        if hasattr(predictor, "features"):
            expected = set(predictor.features)
            available = set(df.columns)
            missing = expected - available
            if missing:
                print(f"Stuff model missing features: {missing}")
            extra = available & expected
            print(
                f"Stuff model using {len(extra)} of {len(expected)} expected features"
            )
        preds = predictor.predict(df)
    except Exception as e:
        print(f"Stuff model prediction failed: {e}")
        return None
    return np.asarray(preds)


def main(
    min_season: int,
    max_season: int,
    out_dir: Path,
    level_ids: list[int],
    save_parquet: bool = True,
) -> None:
    """Pull pitch data, apply models, and save to parquet for aggregation."""
    cfg = load_db_config()
    models = load_models()
    pitch = read_pitch_data(cfg, min_season, max_season, level_ids)
    pitch = add_baseout(cfg, pitch)
    pitch = add_features(pitch)
    pitch = apply_models(pitch, models)

    out_dir.mkdir(parents=True, exist_ok=True)
    if save_parquet:
        parquet_path = out_dir / f"pitch_data_{min_season}_{max_season}.parquet"
        pitch.write_parquet(parquet_path)
        print(f"Saved {len(pitch):,} pitch rows to {parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull pitch data and apply models")
    parser.add_argument("--min-season", type=int, default=2025)
    parser.add_argument("--max-season", type=int, default=2025)
    parser.add_argument("--level-ids", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.getenv("POOBAH_OUT_DIR", OUT_DIR)),
        help="Output directory for parquet file",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the parquet file",
    )
    args = parser.parse_args()
    main(
        min_season=args.min_season,
        max_season=args.max_season,
        out_dir=args.out_dir,
        level_ids=args.level_ids,
        save_parquet=not args.no_save,
    )
