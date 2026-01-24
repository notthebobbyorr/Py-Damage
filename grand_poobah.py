# -*- coding: utf-8 -*-
"""
Grand Poobah (Python)

Best-effort Python/Polars translation of the R Grand Poobah pipeline.
Creates app-ready CSVs with placeholder columns where model outputs are required.
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

        # Create a separate dataframe for SEAGER prediction with encoded values
        seager_df = df.select(seager_features).to_pandas()

        # Use LabelEncoder for categorical columns
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
            # Fallback if sklearn not available
            if "stands" in seager_features:
                seager_df["stands"] = (
                    seager_df["stands"].map({"L": 0, "R": 1}).fillna(1)
                )
            if "throws" in seager_features:
                seager_df["throws"] = (
                    seager_df["throws"].map({"L": 0, "R": 1}).fillna(1)
                )

        # Cast numeric columns
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
        # Debug: check which features the model expects vs what we have
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


def compute_stuff_stats(
    pitch_df: pl.DataFrame,
    group_cols: Iterable[str],
    raw_col: str = "stuff_raw",
) -> pl.DataFrame:
    if pitch_df.is_empty() or raw_col not in pitch_df.columns:
        return pl.DataFrame()
    return (
        pitch_df.filter(pl.col("level_id") == 1)
        .group_by(list(group_cols))
        .agg(
            [
                pl.col(raw_col).mean().alias("stuff_raw_mean"),
                pl.col(raw_col).std().alias("stuff_raw_std"),
            ]
        )
    )


def add_stuff_grade(
    df: pl.DataFrame,
    stats_df: pl.DataFrame,
    join_cols: Iterable[str],
    raw_col: str = "stuff_raw",
    grade_col: str = "stuff",
    z_col: str = "stuff_z",
) -> pl.DataFrame:
    if df.is_empty() or raw_col not in df.columns or stats_df.is_empty():
        return df
    join_cols = list(join_cols)
    stats_cols = join_cols + ["stuff_raw_mean", "stuff_raw_std"]
    stats_df = stats_df.select([c for c in stats_cols if c in stats_df.columns])
    df = df.join(stats_df, on=join_cols, how="left")
    z_expr = (
        pl.when(pl.col("stuff_raw_std").is_not_null() & (pl.col("stuff_raw_std") > 0))
        .then((pl.col(raw_col) - pl.col("stuff_raw_mean")) / pl.col("stuff_raw_std"))
        .otherwise(pl.lit(None))
    )
    grade_expr = (STUFF_SCALE_MEAN - (STUFF_SCALE_STD * z_expr)).clip(20, 80)
    return df.with_columns([z_expr.alias(z_col), grade_expr.alias(grade_col)]).drop(
        [c for c in ["stuff_raw_mean", "stuff_raw_std"] if c in df.columns]
    )


def _pos_label(pos: int | None) -> str:
    mapping = {
        1: "P",
        2: "C",
        3: "X1B",
        4: "X2B",
        5: "X3B",
        6: "SS",
        7: "OF",
        8: "OF",
        9: "OF",
        10: "UT",
        11: "UT",
        12: "UT",
    }
    if pos is None:
        return "NA"
    return mapping.get(int(pos), "NA")


def build_hitters(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = df.with_columns(
        pl.col("batter_position")
        .map_elements(_pos_label, return_dtype=pl.Utf8)
        .alias("position")
    )

    hitters = (
        df.group_by(["batter_mlbid", "hitter_name", "level_id", "season"])
        .agg(
            [
                pl.len().alias("pitches"),
                pl.n_unique("pa_id").alias("PA"),
                pl.sum("bbe").alias("bbe"),
                pl.quantile("exit_velo", 0.9).alias("EV90th"),
                pl.max("exit_velo").alias("max_EV"),
                (
                    100
                    * (
                        (pl.col("launch_angle") >= 20)
                        & (pl.col("spray_angle_adj") < -15)
                        & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("pull_FB_pct"),
                (
                    100
                    * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                    / (pl.col("is_inzone_pi") == False).sum()
                ).alias("chase"),
                (
                    100
                    * (
                        (pl.col("whiff") != 1)
                        & (pl.col("swing") == 1)
                        & (pl.col("is_inzone_pi") == True)
                    ).sum()
                    / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
                ).alias("z_con"),
                (
                    100
                    * (
                        (pl.col("whiff") == 1) & (pl.col("pi_pitch_group") != "FA")
                    ).sum()
                    / (
                        (pl.col("swing") == 1) & (pl.col("pi_pitch_group") != "FA")
                    ).sum()
                ).alias("secondary_whiff_pct"),
                (
                    100
                    * ((pl.col("whiff") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                    / ((pl.col("swing") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                ).alias("whiffs_vs_95"),
                (
                    100
                    * (
                        (pl.col("is_in_play") == True)
                        & (pl.col("damage_pred").is_not_null())
                        & (pl.col("exit_velo") >= pl.col("damage_pred"))
                        & (pl.col("launch_angle") > 0)
                        & (pl.col("spray_angle_adj") >= -50)
                        & (pl.col("spray_angle_adj") <= 50)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("damage_rate"),
                (
                    100
                    * ((pl.col("decision_value") > 0) & (pl.col("swing") == 0)).sum()
                    / (pl.col("decision_value") > 0).sum()
                ).alias("selection_skill"),
                (
                    100
                    * ((pl.col("decision_value") < 0) & (pl.col("swing") == 0)).sum()
                    / (pl.col("swing") == 0).sum()
                ).alias("hittable_pitches_taken"),
                (
                    100
                    * (
                        pl.when(pl.col("swing") == 1)
                        .then(pl.col("pred_whiff_loc"))
                        .mean()
                        - (pl.col("whiff").sum() / (pl.col("swing") == 1).sum())
                    )
                ).alias("contact_vs_avg"),
                (
                    100
                    * (
                        (pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("LA_lte_0"),
                (
                    100
                    * (
                        (pl.col("launch_angle") >= 0)
                        & (pl.col("launch_angle") <= 20)
                        & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("LD_pct"),
                (
                    100
                    * (
                        (pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("LA_gte_20"),
                pl.mean("bat_speed").alias("bat_speed"),
                pl.mean("swing_length").alias("swing_length"),
                pl.mean("attack_angle").alias("attack_angle"),
                pl.mean("swing_path_tilt").alias("swing_path_tilt"),
                # Collect unique team codes into a pipe-separated string (exclude numeric mlbid fallbacks)
                pl.col("hitting_code")
                .filter(
                    pl.col("hitting_code").is_not_null()
                    & ~pl.col("hitting_code").str.contains(r"^\d+$")
                )
                .unique()
                .sort()
                .implode()
                .list.join(" | ")
                .alias("team"),
            ]
        )
        .with_columns(
            (pl.col("selection_skill") - pl.col("hittable_pitches_taken")).alias(
                "SEAGER"
            )
        )
    )

    pos_counts = (
        df.group_by(
            [
                "batter_mlbid",
                "hitter_name",
                "level_id",
                "season",
                "position",
            ]
        )
        .agg(pl.n_unique("pa_id").alias("PA_pos"))
        .pivot(
            values="PA_pos",
            index=["batter_mlbid", "hitter_name", "level_id", "season"],
            columns="position",
        )
        .fill_null(0)
    )

    hitters = hitters.join(
        pos_counts,
        on=["batter_mlbid", "hitter_name", "level_id", "season"],
        how="left",
    )

    desired_pos = ["UT", "C", "X1B", "X2B", "X3B", "SS", "OF", "P", "NA"]
    for col in desired_pos:
        if col not in hitters.columns:
            hitters = hitters.with_columns(pl.lit(0).alias(col))

    return hitters


def build_pitchers(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    pitchers = df.group_by(
        ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand"]
    ).agg(
        [
            pl.len().alias("pitches"),
            pl.n_unique("pa_id").alias("TBF"),
            (pl.sum("outs_recorded") / 3).round(1).alias("IP"),
            (pl.n_unique("pa_id") / pl.n_unique("game_pk")).alias("TBF_per_G"),
            (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
            ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
            (
                100
                * (
                    (pl.col("whiff") != 1)
                    & (pl.col("swing") == 1)
                    & (pl.col("is_inzone_pi") == True)
                ).sum()
                / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
            ).alias("Z_Contact"),
            (
                100
                * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                / (pl.col("is_inzone_pi") == False).sum()
            ).alias("Chase"),
            (
                100
                * (
                    ((pl.col("whiff") == 1)).sum()
                    + ((pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)).sum()
                )
                / pl.len()
            ).alias("CSW"),
            (100 * pl.col("pred_whiff_base").mean()).alias("pWhiff"),
            (100 * (pl.col("pitch_group") == "FA").sum() / pl.len()).alias("FA_pct"),
            (pl.when(pl.col("pitch_group") == "BR").then(pl.col("rpm")).mean()).alias(
                "BB_rpm"
            ),
            (
                pl.when(pl.col("pitch_group") == "FA")
                .then(pl.col("spin_efficiency"))
                .mean()
            ).alias("FA_spin_eff"),
            (
                100
                * ((pl.col("launch_angle") <= 0) & (pl.col("is_in_play") == True)).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_lte_0"),
            (
                100
                * (
                    (pl.col("launch_angle") >= 0)
                    & (pl.col("launch_angle") <= 20)
                    & (pl.col("is_in_play") == True)
                ).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LD_pct"),
            (
                100
                * (
                    (pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True)
                ).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_gte_20"),
            pl.mean("primary_velo").alias("fastball_velo"),
            pl.max("pitch_velo").alias("max_velo"),
            pl.mean("primary_vaa").alias("fastball_vaa"),
            pl.mean("stuff_raw").alias("stuff_raw"),
            pl.mean("release_z").alias("rel_z"),
            pl.mean("release_x").alias("rel_x"),
            pl.mean("ext").alias("ext"),
            pl.mean("arm_angle").alias("arm_angle"),
            pl.col("primary_tag")
            .filter(pl.col("primary_tag").is_not_null())
            .unique()
            .implode()
            .list.join(", ")
            .alias("primary_pitches"),
            # Collect unique team codes into a pipe-separated string (exclude numeric mlbid fallbacks)
            pl.col("pitching_code")
            .filter(
                pl.col("pitching_code").is_not_null()
                & ~pl.col("pitching_code").str.contains(r"^\d+$")
            )
            .unique()
            .sort()
            .implode()
            .list.join(" | ")
            .alias("team"),
        ]
    )
    return pitchers


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


def build_pitch_types(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _tag_pitch(df)
    pitch_types = df.group_by(
        [
            "name",
            "level_id",
            "pitcher_mlbid",
            "pitcher_hand",
            "season",
            "pitch_tag",
        ]
    ).agg(
        [
            pl.len().alias("pitches"),
            (100 * (pl.len() / pl.sum("pitch_of_ab"))).alias("pct"),
            pl.mean("stuff_raw").alias("stuff_raw"),
            pl.mean("pitch_velo").alias("velo"),
            pl.max("pitch_velo").alias("max_velo"),
            pl.mean("vaa").alias("vaa"),
            pl.mean("haa").alias("haa"),
            pl.mean("vbreak").alias("vbreak"),
            pl.mean("hbreak").alias("hbreak"),
            pl.mean("loc_adj_vaa").alias("loc_adj_vaa"),
            pl.mean("rpm").alias("rpm"),
            pl.mean("axis").alias("axis"),
            pl.mean("spin_efficiency").alias("spin_efficiency"),
            pl.col("primary_tag")
            .filter(pl.col("primary_tag").is_not_null())
            .unique()
            .implode()
            .list.join(", ")
            .alias("primary_pitches"),
            pl.mean("primary_loc_adj_vaa").alias("primary_loc_adj_vaa"),
            pl.mean("primary_velo").alias("primary_velo"),
            pl.mean("primary_rpm").alias("primary_rpm"),
            pl.mean("primary_axis").alias("primary_axis"),
            pl.mean("primary_hbreak").alias("primary_hbreak"),
            pl.mean("primary_vbreak").alias("primary_vbreak"),
            pl.mean("primary_z_release").alias("primary_z_release"),
            pl.mean("primary_x_release").alias("primary_x_release"),
            (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
            ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
            (
                100
                * (
                    (pl.col("whiff") != 1)
                    & (pl.col("swing") == 1)
                    & (pl.col("is_inzone_pi") == True)
                ).sum()
                / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
            ).alias("Z_Contact"),
            (
                100
                * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                / (pl.col("is_inzone_pi") == False).sum()
            ).alias("Chase"),
            (
                100
                * (
                    ((pl.col("whiff") == 1)).sum()
                    + ((pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)).sum()
                )
                / pl.len()
            ).alias("CSW"),
            (
                100
                * pl.when(pl.col("swing") == 1).then(pl.col("pred_whiff_base")).mean()
            ).alias("pred_whiff_pct"),
            # Collect unique team codes into a pipe-separated string (exclude numeric mlbid fallbacks)
            pl.col("pitching_code")
            .filter(
                pl.col("pitching_code").is_not_null()
                & ~pl.col("pitching_code").str.contains(r"^\d+$")
            )
            .unique()
            .sort()
            .implode()
            .list.join(" | ")
            .alias("team"),
        ]
    )
    return pitch_types


def build_team_hitting(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    team = (
        df.group_by(["hitting_code", "level_id", "season"])
        .agg(
            [
                pl.n_unique("pa_id").alias("PA"),
                pl.sum("bbe").alias("bbe"),
                pl.quantile("exit_velo", 0.9).alias("EV90th"),
                (
                    100
                    * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                    / (pl.col("is_inzone_pi") == False).sum()
                ).alias("chase"),
                (
                    100
                    * (
                        (pl.col("whiff") != 1)
                        & (pl.col("swing") == 1)
                        & (pl.col("is_inzone_pi") == True)
                    ).sum()
                    / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
                ).alias("z_con"),
                (
                    100
                    * (
                        (pl.col("whiff") == 1) & (pl.col("pi_pitch_group") != "FA")
                    ).sum()
                    / (
                        (pl.col("swing") == 1) & (pl.col("pi_pitch_group") != "FA")
                    ).sum()
                ).alias("secondary_whiff_pct"),
                (
                    100
                    * (
                        (pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("LA_lte_0"),
                (
                    100
                    * (
                        (pl.col("launch_angle") >= 0)
                        & (pl.col("launch_angle") <= 20)
                        & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("LD_pct"),
                (
                    100
                    * (
                        (pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("LA_gte_20"),
                (
                    100
                    * (
                        (pl.col("launch_angle") >= 20)
                        & (pl.col("spray_angle_adj") < -15)
                        & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("pull_FB_pct"),
                (
                    100
                    * (
                        (pl.col("is_in_play") == True)
                        & (pl.col("damage_pred").is_not_null())
                        & (pl.col("exit_velo") >= pl.col("damage_pred"))
                        & (pl.col("launch_angle") > 0)
                        & (pl.col("spray_angle_adj") >= -50)
                        & (pl.col("spray_angle_adj") <= 50)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("damage_rate"),
                (
                    100
                    * ((pl.col("decision_value") > 0) & (pl.col("swing") == 0)).sum()
                    / (pl.col("decision_value") > 0).sum()
                ).alias("selection_skill"),
                (
                    100
                    * ((pl.col("decision_value") < 0) & (pl.col("swing") == 0)).sum()
                    / (pl.col("swing") == 0).sum()
                ).alias("hittable_pitches_taken"),
                (
                    100
                    * (
                        pl.when(pl.col("swing") == 1)
                        .then(pl.col("pred_whiff_loc"))
                        .mean()
                        - (pl.col("whiff").sum() / (pl.col("swing") == 1).sum())
                    )
                ).alias("contact_vs_avg"),
            ]
        )
        .with_columns(
            (pl.col("selection_skill") - pl.col("hittable_pitches_taken")).alias(
                "SEAGER"
            )
        )
    )
    return team


def build_team_pitching(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    team = df.group_by(["pitching_code", "level_id", "season"]).agg(
        [
            pl.n_unique("pa_id").alias("TBF"),
            (pl.sum("outs_recorded") / 3).round(1).alias("IP"),
            pl.sum("bbe").alias("bbe"),
            pl.len().alias("pitches"),
            pl.mean("stuff_raw").alias("stuff_raw"),
            (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
            ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
            (
                100
                * (
                    (pl.col("whiff") != 1)
                    & (pl.col("swing") == 1)
                    & (pl.col("is_inzone_pi") == True)
                ).sum()
                / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
            ).alias("Z_Contact"),
            (
                100
                * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                / (pl.col("is_inzone_pi") == False).sum()
            ).alias("Chase"),
            (
                100
                * (
                    ((pl.col("whiff") == 1)).sum()
                    + ((pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)).sum()
                )
                / pl.len()
            ).alias("CSW"),
            (100 * pl.col("pred_whiff_base").mean()).alias("pWhiff"),
            (
                100
                * ((pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True)).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_lte_0"),
            (
                100
                * (
                    (pl.col("launch_angle") >= 0)
                    & (pl.col("launch_angle") <= 20)
                    & (pl.col("is_in_play") == True)
                ).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LD_pct"),
            (
                100
                * (
                    (pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True)
                ).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_gte_20"),
            pl.mean("primary_velo").alias("fastball_velo"),
            pl.mean("primary_vaa").alias("fastball_vaa"),
        ]
    )
    return team


def add_percentiles(
    df: pl.DataFrame,
    group_cols: Iterable[str],
    value_cols: Iterable[str],
    filter_col: str | None = None,
    min_threshold: float | None = None,
) -> pl.DataFrame:
    """Add percentile columns for value_cols, grouped by group_cols.

    If filter_col and min_threshold are provided, percentiles are computed only
    among rows meeting the threshold, but results are returned for all rows.
    """
    if df.is_empty():
        return df
    df_pd = df.to_pandas()
    group_list = list(group_cols)
    value_list = list(value_cols)

    # If threshold specified, compute percentiles only among qualified rows
    if filter_col and min_threshold is not None and filter_col in df_pd.columns:
        qualified = df_pd[df_pd[filter_col] >= min_threshold].copy()
        if qualified.empty:
            # No qualified rows, return df with null percentile columns
            for col in value_list:
                df_pd[f"{col}_pctile"] = None
            return pl.from_pandas(df_pd)

        # Compute percentiles among qualified rows
        pct = qualified.groupby(group_list)[value_list].rank(pct=True) * 100
        pct.columns = [f"{c}_pctile" for c in pct.columns]
        pct.index = qualified.index

        # Merge back to original dataframe (unqualified rows get NaN)
        for col in pct.columns:
            df_pd[col] = pct[col]
    else:
        pct = df_pd.groupby(group_list)[value_list].rank(pct=True) * 100
        pct.columns = [f"{c}_pctile" for c in pct.columns]
        pct = pct.reset_index(drop=True)
        df_pd = pd.concat([df_pd.reset_index(drop=True), pct], axis=1)

    return pl.from_pandas(df_pd)


def write_csv(df: pl.DataFrame, name: str, out_dir: Path) -> None:
    path = out_dir / name
    df.to_pandas().to_csv(path, index=False)


def compute_stuff_percentiles(
    df: pl.DataFrame,
    raw_col: str = "stuff_raw",
    min_pitches: int = 50,
) -> pl.DataFrame:
    """Compute stuff grade percentile thresholds by season + pitch_tag.

    Uses percentile-based scaling where:
    - 1st percentile of stuff_raw (best) maps to grade 80
    - 99th percentile of stuff_raw (worst) maps to grade 20

    Percentiles are computed from pitcher-level averages (min pitches) to avoid
    high-volume pitchers skewing the distribution.

    For early-season data where a pitch type doesn't have enough qualified pitchers,
    falls back to the average percentiles from other seasons for that pitch type.

    Returns a DataFrame with columns: season, pitch_tag, stuff_p01, stuff_p99
    """
    if df.is_empty() or raw_col not in df.columns:
        return pl.DataFrame()

    # First, compute pitcher-level averages for each pitch type (MLB only, min pitches)
    pitcher_avgs = (
        df.filter(pl.col("level_id") == 1)
        .group_by(["season", "pitcher_mlbid", "pitch_tag"])
        .agg(
            [
                pl.col(raw_col).mean().alias("pitcher_stuff_avg"),
                pl.len().alias("pitch_count"),
            ]
        )
        .filter(pl.col("pitch_count") >= min_pitches)
    )

    # Compute percentiles from the pitcher-level averages by season
    stats = pitcher_avgs.group_by(["season", "pitch_tag"]).agg(
        [
            pl.col("pitcher_stuff_avg").quantile(0.01).alias("stuff_p01"),
            pl.col("pitcher_stuff_avg").quantile(0.99).alias("stuff_p99"),
            pl.len().alias("n_pitchers"),
        ]
    )

    # Compute fallback percentiles averaged across all seasons for each pitch type
    fallback_stats = pitcher_avgs.group_by(["pitch_tag"]).agg(
        [
            pl.col("pitcher_stuff_avg").quantile(0.01).alias("fallback_p01"),
            pl.col("pitcher_stuff_avg").quantile(0.99).alias("fallback_p99"),
        ]
    )

    # Join both and use fallback when current season has insufficient data (< 10 pitchers)
    stats = stats.join(fallback_stats, on=["pitch_tag"], how="left")
    stats = stats.with_columns(
        [
            pl.when(pl.col("n_pitchers") >= 10)
            .then(pl.col("stuff_p01"))
            .otherwise(pl.col("fallback_p01"))
            .alias("stuff_p01"),
            pl.when(pl.col("n_pitchers") >= 10)
            .then(pl.col("stuff_p99"))
            .otherwise(pl.col("fallback_p99"))
            .alias("stuff_p99"),
        ]
    ).drop(["fallback_p01", "fallback_p99", "n_pitchers"])

    return stats


def apply_stuff_grade(
    df: pl.DataFrame,
    percentiles: pl.DataFrame,
    raw_col: str = "stuff_raw",
    grade_col: str = "stuff",
) -> pl.DataFrame:
    """Apply stuff grades to aggregated data using precomputed percentiles.

    Joins percentile thresholds and computes grades from the aggregated stuff_raw.
    Linear interpolation: p01 -> 80, p99 -> 20, clipped to 20-80 range.

    Note: stuff_raw is inverted (lower = better), so p01 is best and p99 is worst.
    """
    if df.is_empty() or raw_col not in df.columns or percentiles.is_empty():
        return df

    df = df.join(percentiles, on=["season", "pitch_tag"], how="left")
    grade_expr = (
        pl.when(
            pl.col("stuff_p99").is_not_null()
            & pl.col("stuff_p01").is_not_null()
            & (pl.col("stuff_p99") != pl.col("stuff_p01"))
        )
        .then(
            (
                80.0
                - 60.0
                * (pl.col(raw_col) - pl.col("stuff_p01"))
                / (pl.col("stuff_p99") - pl.col("stuff_p01"))
            )
            .clip(20, 80)
            .round(0)
            .cast(pl.Int64)
        )
        .otherwise(pl.lit(None))
    )
    return df.with_columns([grade_expr.alias(grade_col)]).drop(
        ["stuff_p01", "stuff_p99"]
    )


# ARCHIVED: Raw pitch-level percentile method (may skew toward high-volume pitchers)
# def add_pitch_level_stuff_grade_raw_percentiles(
#     df: pl.DataFrame,
#     raw_col: str = "stuff_raw",
#     grade_col: str = "stuff",
# ) -> pl.DataFrame:
#     """Add stuff grades at pitch level, standardized by season + pitch_tag.
#
#     Uses percentile-based scaling where:
#     - 20th percentile of stuff_raw (best) maps to grade 80
#     - 80th percentile of stuff_raw (worst) maps to grade 20
#
#     Note: stuff_raw is inverted (lower = better), so p20 is best and p80 is worst.
#     Grades are clipped to 20-80 range.
#     """
#     if df.is_empty() or raw_col not in df.columns:
#         return df
#     # Compute percentiles grouped by season + pitch_tag (MLB only)
#     stats = (
#         df.filter(pl.col("level_id") == 1)
#         .group_by(["season", "pitch_tag"])
#         .agg(
#             [
#                 pl.col(raw_col).quantile(0.20).alias("stuff_p20"),
#                 pl.col(raw_col).quantile(0.80).alias("stuff_p80"),
#             ]
#         )
#     )
#     df = df.join(stats, on=["season", "pitch_tag"], how="left")
#     # Linear interpolation: p20 -> 80, p80 -> 20
#     # grade = 80 - 60 * (raw - p20) / (p80 - p20)
#     grade_expr = (
#         pl.when(
#             pl.col("stuff_p80").is_not_null()
#             & pl.col("stuff_p20").is_not_null()
#             & (pl.col("stuff_p80") != pl.col("stuff_p20"))
#         )
#         .then(
#             (
#                 80.0
#                 - 60.0
#                 * (pl.col(raw_col) - pl.col("stuff_p20"))
#                 / (pl.col("stuff_p80") - pl.col("stuff_p20"))
#             ).clip(20, 80)
#         )
#         .otherwise(pl.lit(None))
#     )
#     return df.with_columns([grade_expr.alias(grade_col)]).drop(
#         ["stuff_p20", "stuff_p80"]
#     )


def main(min_season: int, max_season: int, out_dir: Path, level_ids: list[int]) -> None:
    cfg = load_db_config()
    models = load_models()
    pitch = read_pitch_data(cfg, min_season, max_season, level_ids)
    pitch = add_baseout(cfg, pitch)
    pitch = add_features(pitch)
    pitch = apply_models(pitch, models)

    # Compute stuff grade percentiles from pitcher-level averages (by season + pitch_tag)
    stuff_percentiles = compute_stuff_percentiles(pitch, min_pitches=50)

    if os.getenv("POOBAH_SAMPLE_ROWS"):
        try:
            sample_rows = int(os.getenv("POOBAH_SAMPLE_ROWS", "0"))
        except ValueError:
            sample_rows = 0
        if sample_rows != 0:
            sample_path = out_dir / f"pitch_data_{min_season}_{max_season}.parquet"
            if sample_rows < 0:
                # Save all rows
                pitch.write_parquet(sample_path)
                print(f"Saved all {len(pitch):,} pitch rows to {sample_path}")
            else:
                # Save sample
                pitch.head(sample_rows).write_parquet(sample_path)
                print(f"Saved {sample_rows:,} sample pitch rows to {sample_path}")

    hitters = build_hitters(pitch)
    pitchers = build_pitchers(pitch)
    pitch_types = build_pitch_types(pitch)
    team_hitting = build_team_hitting(pitch)
    team_pitching = build_team_pitching(pitch)

    # Apply stuff grades to pitch_types (has pitch_tag for direct grading)
    pitch_types = apply_stuff_grade(pitch_types, stuff_percentiles)

    # For pitchers, compute weighted average stuff grade from pitch types
    pitcher_stuff = (
        pitch_types.group_by(["pitcher_mlbid", "season", "level_id"])
        .agg((pl.col("stuff") * pl.col("pitches")).sum() / pl.col("pitches").sum())
        .rename({"stuff": "stuff_grade"})
    )
    pitchers = (
        pitchers.join(
            pitcher_stuff, on=["pitcher_mlbid", "season", "level_id"], how="left"
        )
        .with_columns(pl.col("stuff_grade").alias("stuff"))
        .drop("stuff_grade")
    )

    # For team_pitching, compute stuff grades from raw pitch data aggregated by team
    # First aggregate stuff_raw by pitching_code + pitch_tag, then apply grades
    team_pitch_types = (
        _tag_pitch(pitch)
        .group_by(["pitching_code", "season", "level_id", "pitch_tag"])
        .agg(
            [
                pl.mean("stuff_raw").alias("stuff_raw"),
                pl.len().alias("pitches"),
            ]
        )
    )
    team_pitch_types = apply_stuff_grade(team_pitch_types, stuff_percentiles)
    team_stuff = (
        team_pitch_types.group_by(["pitching_code", "season", "level_id"])
        .agg((pl.col("stuff") * pl.col("pitches")).sum() / pl.col("pitches").sum())
        .rename({"stuff": "stuff_grade"})
    )
    team_pitching = (
        team_pitching.join(
            team_stuff, on=["pitching_code", "season", "level_id"], how="left"
        )
        .with_columns(pl.col("stuff_grade").alias("stuff"))
        .drop("stuff_grade")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(hitters, f"damage_pos_{min_season}_{max_season}.csv", out_dir)
    write_csv(pitchers, "pitcher_stuff_new.csv", out_dir)
    write_csv(pitch_types, "new_pitch_types.csv", out_dir)
    write_csv(team_hitting, "new_team_damage.csv", out_dir)
    write_csv(team_pitching, "new_team_stuff.csv", out_dir)

    hitter_pct = add_percentiles(
        hitters,
        group_cols=["season", "level_id"],
        value_cols=[
            "SEAGER",
            "selection_skill",
            "hittable_pitches_taken",
            "damage_rate",
            "EV90th",
            "max_EV",
            "pull_FB_pct",
            "chase",
            "z_con",
            "secondary_whiff_pct",
            "contact_vs_avg",
        ],
        filter_col="PA",
        min_threshold=200,
    )
    write_csv(hitter_pct, "hitter_pctiles.csv", out_dir)

    pitcher_pct = add_percentiles(
        pitchers,
        group_cols=["season", "level_id"],
        value_cols=[
            "stuff",
            "fastball_velo",
            "max_velo",
            "fastball_vaa",
            "SwStr",
            "Ball_pct",
            "Z_Contact",
            "Chase",
            "CSW",
            "rel_z",
            "rel_x",
            "ext",
        ],
        filter_col="IP",
        min_threshold=40,
    )
    write_csv(pitcher_pct, "pitcher_pctiles.csv", out_dir)

    pitch_types_pct = add_percentiles(
        pitch_types,
        group_cols=["season", "level_id", "pitch_tag"],
        value_cols=[
            "pct",
            "stuff",
            "velo",
            "max_velo",
            "vaa",
            "haa",
            "vbreak",
            "hbreak",
            "SwStr",
            "Ball_pct",
            "Z_Contact",
            "Chase",
            "CSW",
        ],
        filter_col="pitches",
        min_threshold=100,
    )
    write_csv(pitch_types_pct, "pitch_types_pctiles.csv", out_dir)

    write_csv(team_hitting, "new_hitting_lg_avg.csv", out_dir)
    write_csv(team_pitching, "new_lg_stuff.csv", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-season", type=int, default=2025)
    parser.add_argument("--max-season", type=int, default=2025)
    parser.add_argument("--level-ids", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.getenv("POOBAH_OUT_DIR", OUT_DIR)),
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Write pitch-level CSV: -1 for all rows, N>0 for sample of N rows, 0 to skip",
    )
    args = parser.parse_args()
    if args.sample_rows != 0:
        os.environ["POOBAH_SAMPLE_ROWS"] = str(args.sample_rows)
    main(
        min_season=args.min_season,
        max_season=args.max_season,
        out_dir=args.out_dir,
        level_ids=args.level_ids,
    )
