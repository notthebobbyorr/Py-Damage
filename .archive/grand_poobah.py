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
    from catboost import CatBoostClassifier
except ImportError:  # optional dependency
    CatBoostClassifier = None

try:
    from pygam import LinearGAM
except ImportError:  # optional dependency
    LinearGAM = None

try:
    from joblib import load as joblib_load
except ImportError:  # optional dependency
    joblib_load = None

import pickle


DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR
MODEL_DIR = DATA_DIR


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
WHIFF_BASE_NUM_FEATURES = [feat for feat in WHIFF_NUM_FEATURES if feat not in {"x", "z"}]


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

    return bundle


def _ensure_columns(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return df
    return df.with_columns([pl.lit(None).alias(c) for c in missing])


def _predict_model(model: object, frame: pl.DataFrame, feature_cols: list[str]) -> np.ndarray | None:
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


def read_pitch_data(cfg: DbConfig, min_season: int, max_season: int, level_ids: Iterable[int]) -> pl.DataFrame:
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
        print(f"Running pitch query for seasons {min_season}-{max_season} and levels {level_clause}...")
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
            .then(-1 * (pl.arctan2(pl.col("hc_x_adj"), pl.col("hc_y_adj")) * (180 / np.pi) * 0.75))
            .otherwise(pl.arctan2(pl.col("hc_x_adj"), pl.col("hc_y_adj")) * (180 / np.pi) * 0.75)
            .alias("spray_angle_adj"),
            (pl.col("batter_name_first") + pl.lit(" ") + pl.col("batter_name_last")).alias("hitter_name"),
            (pl.col("pitcher_name_first") + pl.lit(" ") + pl.col("pitcher_name_last")).alias("pitcher_name"),
            (pl.col("pitcher_name_first") + pl.lit(" ") + pl.col("pitcher_name_last")).alias("name"),
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
            .when(pl.col("pi_pitch_group").is_in(["CH", "FS"]))
            .then(pl.lit("OFF"))
            .otherwise(pl.lit("OTHER"))
            .alias("pitch_group"),
            pl.when(pl.col("batter_hand") == "L")
            .then(-1 * pl.col("pfx_x_short"))
            .otherwise(pl.col("pfx_x_short"))
            .alias("pfx_x_short_adj"),
            pl.when(pl.col("half_inning") == "bottom")
            .then(pl.col("home_team"))
            .otherwise(pl.col("away_team"))
            .alias("hitting_code"),
            pl.when(pl.col("half_inning") == "bottom")
            .then(pl.col("away_team"))
            .otherwise(pl.col("home_team"))
            .alias("pitching_code"),
            pl.col("balls_before").alias("balls"),
            pl.col("strikes_before").alias("strikes"),
            pl.col("pitcher_hand").alias("throws"),
            pl.col("batter_hand").alias("stands"),
            (pl.col('outs_end') - pl.col('outs_start')).alias('outs_recorded')
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("swing_type") == "swing").then(1).otherwise(0).alias("swing"),
            pl.when(pl.col("swing_type") == "swing").then(1).otherwise(0).alias("is_swing"),
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
            (1.8 * pl.col("pfx_x_short_adj")).alias("hbreak"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("game_pk").cast(str) + "_" + pl.col("at_bat_index").cast(str)).alias("pa_id"),
        ]
    )

    k_const = 5.383e-03
    z_const = 32.174
    yR = 60.5 - pl.col("ext")
    disc = (pl.col("vy0") ** 2) - (2 * pl.col("ay") * (50 - yR))
    tR = pl.when((pl.col("ay") != 0) & (disc >= 0)).then(
        (-pl.col("vy0") - disc.sqrt()) / pl.col("ay")
    )
    vxR = pl.col("vx0") + (pl.col("ax") * tR)
    vyR = pl.col("vy0") + (pl.col("ay") * tR)
    vzR = pl.col("vz0") + (pl.col("az") * tR)
    disc_tf = (vyR ** 2) - (2 * pl.col("ay") * (yR - (17 / 12)))
    tf = pl.when((pl.col("ay") != 0) & (disc_tf >= 0)).then((-vyR - disc_tf.sqrt()) / pl.col("ay"))
    vxbar = (2 * vxR + (pl.col("ax") * tf)) / 2
    vybar = (2 * vyR + (pl.col("ay") * tf)) / 2
    vzbar = (2 * vzR + (pl.col("az") * tf)) / 2
    vbar = (vxbar**2 + vybar**2 + vzbar**2).sqrt()
    adrag = -(pl.col("ax") * vxbar + pl.col("ay") * vybar + (pl.col("az") + z_const) * vzbar) / vbar
    amagx = pl.col("ax") + adrag * vxbar / vbar
    amagy = pl.col("ay") + adrag * vybar / vbar
    amagz = pl.col("az") + adrag * vzbar / vbar + z_const
    amag = (amagx**2 + amagy**2 + amagz**2).sqrt()
    cl = amag / (k_const * vbar**2)
    s_val = 0.166 * (0.336 / (0.336 - cl)).log()
    spin_t = 78.92 * s_val * vbar
    spin_eff = (0.1 + (spin_t / pl.col("rpm"))).clip(0, 1)

    df = df.with_columns(
        [
            pl.lit(None).cast(pl.Float64).alias("xwt"),
            pl.lit(None).cast(pl.Float64).alias("xgb_woba"),
            pl.lit(None).cast(pl.Float64).alias("damage_pred"),
            pl.lit(None).cast(pl.Float64).alias("decision_value"),
            spin_eff.alias("spin_efficiency"),
            pl.lit(None).cast(pl.Float64).alias("pred_whiff_loc"),
            pl.lit(None).cast(pl.Float64).alias("pred_whiff_base"),
        ]
    )
    return df


def apply_vaa_gam(df: pl.DataFrame, model: object | None) -> pl.DataFrame:
    if df.is_empty():
        return df
    if model is None:
        return df.with_columns(pl.col("vaa").alias("loc_adj_vaa"))
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = ["x", "z", "vaa"]
    preds = _predict_model(model, df, feature_cols)
    if preds is None:
        return df.with_columns(pl.col("vaa").alias("loc_adj_vaa"))
    return df.with_columns((pl.col("vaa") - pl.Series("vaa_pred", preds)).alias("loc_adj_vaa"))


def add_pitcher_context(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    averages = (
        df.group_by(["pitcher_mlbid", "level_id", "season"])
        .agg(
            [
                pl.mean("release_z").alias("avg_release_z"),
                pl.mean("release_x").alias("avg_release_x"),
                pl.mean("ext").alias("avg_ext"),
                pl.mean("arm_angle").alias("arm_angle"),
            ]
        )
    )
    return df.join(averages, on=["pitcher_mlbid", "level_id", "season"], how="left")


def add_primary_pitch_context(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _tag_pitch(df)
    counts = (
        df.filter(pl.col("pitch_tag").is_in(["FA", "SI", "HC"]))
        .group_by(["pitcher_mlbid", "level_id", "season", "pitch_tag"])
        .agg(pl.len().alias("pitch_count"))
        .sort(["pitcher_mlbid", "level_id", "season", "pitch_count"], descending=[False, False, False, True])
    )
    primary = (
        counts.group_by(["pitcher_mlbid", "level_id", "season"])
        .agg(pl.first("pitch_tag").alias("primary_tag"))
    )
    df = df.join(primary, on=["pitcher_mlbid", "level_id", "season"], how="left")
    primary_stats = (
        df.filter(pl.col("pitch_tag") == pl.col("primary_tag"))
        .group_by(["pitcher_mlbid", "level_id", "season"])
        .agg(
            [
                pl.mean("pitch_velo").alias("primary_velo"),
                pl.mean("loc_adj_vaa").alias("primary_loc_adj_vaa"),
                pl.mean("release_z").alias("primary_z_release"),
                pl.mean("release_x").alias("primary_x_release"),
                pl.mean("rpm").alias("primary_rpm"),
                pl.mean("axis").alias("primary_axis"),
            ]
        )
    )
    return df.join(primary_stats, on=["pitcher_mlbid", "level_id", "season"], how="left")


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

    loc_preds = predict_catboost_probs(models.whiff_loc, df, WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES, WHIFF_CAT_FEATURES)
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
            df = df.with_columns(
                [
                    pl.when(pl.col("stands") == "L").then(0).when(pl.col("stands") == "R").then(1).otherwise(None).alias("stands"),
                    pl.col("balls").cast(pl.Int64),
                    pl.col("strikes").cast(pl.Int64),
                ]
            )
            seager_preds = _predict_model(models.seager, df, seager_features)
            if seager_preds is not None:
                df = df.with_columns(pl.Series(name="decision_value_raw", values=seager_preds))
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
            if hasattr(models.seager, "feature_names_in_"):
                seager_features = list(models.seager.feature_names_in_)
            else:
                seager_features = ["balls", "strikes", "x", "z", "pitch_velo", "vaa", "haa", "stands", "throws"]
            print(f"SEAGER model features: {seager_features}")
            if "stands" in seager_features:
                df = df.with_columns(
                    pl.when(pl.col("stands") == "L").then(0).when(pl.col("stands") == "R").then(1).otherwise(None).alias("stands")
                )
            seager_preds = _predict_model(models.seager, df, seager_features)
            if seager_preds is not None:
                df = df.with_columns(pl.Series(name="decision_value_raw", values=seager_preds))
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

    return df

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
        df.group_by(["batter_mlbid", "hitter_name", "level_id", "hitting_code", "season"])
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
                        (pl.col("launch_angle") > 20)
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
                    * ((pl.col("whiff") != 1) & (pl.col("swing") == 1) & (pl.col("is_inzone_pi") == True)).sum()
                    / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
                ).alias("z_con"),
                (
                    100
                    * ((pl.col("whiff") == 1) & (pl.col("pi_pitch_group") != "FA")).sum()
                    / ((pl.col("swing") == 1) & (pl.col("pi_pitch_group") != "FA")).sum()
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
                        pl.when(pl.col("swing") == 1).then(pl.col("pred_whiff_loc")).mean()
                        - (pl.col("whiff").sum() / (pl.col("swing") == 1).sum())
                    )
                ).alias("contact_vs_avg"),
                (
                    100
                    * ((pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("GB_pct"),
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
                    * ((pl.col("launch_angle") > 20) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("FB_pct"),
                pl.mean("bat_speed").alias("bat_speed"),
                pl.mean("swing_length").alias("swing_length"),
                pl.mean("attack_angle").alias("attack_angle"),
                pl.mean("swing_path_tilt").alias("swing_path_tilt"),
            ]
        )
        .with_columns((pl.col("selection_skill") - pl.col("hittable_pitches_taken")).alias("SEAGER"))
    )

    pos_counts = (
        df.group_by(["batter_mlbid", "hitter_name", "level_id", "hitting_code", "season", "position"])
        .agg(pl.n_unique("pa_id").alias("PA_pos"))
        .pivot(values="PA_pos", index=["batter_mlbid", "hitter_name", "level_id", "hitting_code", "season"], columns="position")
        .fill_null(0)
    )

    hitters = hitters.join(pos_counts, on=["batter_mlbid", "hitter_name", "level_id", "hitting_code", "season"], how="left")

    desired_pos = ["UT", "C", "X1B", "X2B", "X3B", "SS", "OF", "P", "NA"]
    for col in desired_pos:
        if col not in hitters.columns:
            hitters = hitters.with_columns(pl.lit(0).alias(col))

    return hitters


def build_pitchers(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    pitchers = (
        df.group_by(["pitcher_mlbid", "name", "season", "level_id", "pitching_code", "pitcher_hand"])
        .agg(
            [
                pl.len().alias("pitches"),
                pl.n_unique("pa_id").alias("TBF"),
                (pl.n_unique("pa_id") / pl.n_unique("game_pk")).alias("TBF_per_G"),
                (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
                ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
                (
                    100
                    * ((pl.col("whiff") != 1) & (pl.col("swing") == 1) & (pl.col("is_inzone_pi") == True)).sum()
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
                    100 * pl.col("pred_whiff_base").mean()
                ).alias("pSwStr"),
                (
                    100 * (pl.col("pitch_group") == "FA").sum() / pl.len()
                ).alias("FA_pct"),
                (
                    pl.when(pl.col("pitch_group") == "BR")
                    .then(pl.col("rpm"))
                    .mean()
                ).alias("BB_rpm"),
                (
                    pl.when(pl.col("pitch_group") == "FA")
                    .then(pl.col("spin_efficiency"))
                    .mean()
                ).alias("FA_spin_eff"),
                (
                    100
                    * ((pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("GB_pct"),
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
                    * ((pl.col("launch_angle") > 20) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("FB_pct"),
                pl.mean("pitch_velo").alias("fastball_velo"),
                pl.max("pitch_velo").alias("max_velo"),
                pl.mean("vaa").alias("fastball_vaa"),
                pl.mean("release_z").alias("rel_z"),
                pl.mean("release_x").alias("rel_x"),
                pl.mean("ext").alias("ext"),
                pl.mean("arm_angle").alias("arm_angle"),
            ]
        )
        .with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("IP"),
                pl.lit(None).cast(pl.Float64).alias("std.ZQ"),
                pl.lit(None).cast(pl.Float64).alias("std.DMG"),
                pl.lit(None).cast(pl.Float64).alias("std.NRV"),
            ]
        )
    )
    return pitchers


def _tag_pitch(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("pi_pitch_sub_type") == "SW")
        .then(pl.lit("SW"))
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
    pitch_types = (
        df.group_by(["name", "level_id", "pitcher_mlbid", "pitcher_hand", "pitching_code", "season", "pitch_tag"])
        .agg(
            [
                pl.len().alias("pitches"),
                (pl.len() / pl.sum("pitch_of_ab")).alias("pct"),
                pl.mean("pitch_velo").alias("velo"),
                pl.max("pitch_velo").alias("max_velo"),
                pl.mean("vaa").alias("vaa"),
                pl.mean("haa").alias("haa"),
                pl.mean("ivb").alias("ivb"),
                pl.mean("hb_arm").alias("hb"),
                (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
                ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
                (
                    100
                    * ((pl.col("whiff") != 1) & (pl.col("swing") == 1) & (pl.col("is_inzone_pi") == True)).sum()
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
                    100 * pl.when(pl.col("swing") == 1).then(pl.col("pred_whiff_base")).mean()
                ).alias("pred_whiff_pct"),
            ]
        )
        .with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("std.ZQ"),
                pl.lit(None).cast(pl.Float64).alias("std.DMG"),
                pl.lit(None).cast(pl.Float64).alias("std.NRV"),
            ]
        )
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
                    * ((pl.col("whiff") != 1) & (pl.col("swing") == 1) & (pl.col("is_inzone_pi") == True)).sum()
                    / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
                ).alias("z_con"),
                (
                    100
                    * ((pl.col("whiff") == 1) & (pl.col("pi_pitch_group") != "FA")).sum()
                    / ((pl.col("swing") == 1) & (pl.col("pi_pitch_group") != "FA")).sum()
                ).alias("secondary_whiff_pct"),
                (
                    100
                    * ((pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("GB_pct"),
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
                    * ((pl.col("launch_angle") > 20) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("FB_pct"),
                (
                    100
                    * (
                        (pl.col("launch_angle") > 20)
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
                        pl.when(pl.col("swing") == 1).then(pl.col("pred_whiff_loc")).mean()
                        - (pl.col("whiff").sum() / (pl.col("swing") == 1).sum())
                    )
                ).alias("contact_vs_avg"),
            ]
        )
        .with_columns((pl.col("selection_skill") - pl.col("hittable_pitches_taken")).alias("SEAGER"))
    )
    return team


def build_team_pitching(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    team = (
        df.group_by(["pitching_code", "level_id", "season"])
        .agg(
            [
                pl.n_unique("pa_id").alias("TBF"),
                pl.sum("bbe").alias("bbe"),
                pl.len().alias("pitches"),
                (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
                ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
                (
                    100
                    * ((pl.col("whiff") != 1) & (pl.col("swing") == 1) & (pl.col("is_inzone_pi") == True)).sum()
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
                    100 * pl.col("pred_whiff_base").mean()
                ).alias("pSwStr"),
                (
                    100
                    * ((pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("GB_pct"),
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
                    * ((pl.col("launch_angle") > 20) & (pl.col("is_in_play") == True)).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("FB_pct"),
                pl.mean("pitch_velo").alias("fastball_velo"),
                pl.mean("vaa").alias("fastball_vaa"),
            ]
        )
        .with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("IP"),
                pl.lit(None).cast(pl.Float64).alias("std.ZQ"),
                pl.lit(None).cast(pl.Float64).alias("std.DMG"),
                pl.lit(None).cast(pl.Float64).alias("std.NRV"),
            ]
        )
    )
    return team


def add_percentiles(df: pl.DataFrame, group_cols: Iterable[str], value_cols: Iterable[str]) -> pl.DataFrame:
    if df.is_empty():
        return df
    df_pd = df.to_pandas()
    pct = df_pd.groupby(list(group_cols))[list(value_cols)].rank(pct=True) * 100
    pct.columns = [f"{c}_pctile" for c in pct.columns]
    pct = pct.reset_index(drop=True)
    merged = pd.concat([df_pd.reset_index(drop=True), pct], axis=1)
    return pl.from_pandas(merged)


def write_csv(df: pl.DataFrame, name: str, out_dir: Path) -> None:
    path = out_dir / name
    df.to_pandas().to_csv(path, index=False)


def main(min_season: int, max_season: int, out_dir: Path, level_ids: list[int]) -> None:
    cfg = load_db_config()
    models = load_models()
    pitch = read_pitch_data(cfg, min_season, max_season, level_ids)
    pitch = add_baseout(cfg, pitch)
    pitch = add_features(pitch)
    pitch = apply_models(pitch, models)
    if os.getenv("POOBAH_SAMPLE_ROWS"):
        try:
            sample_rows = int(os.getenv("POOBAH_SAMPLE_ROWS", "0"))
        except ValueError:
            sample_rows = 0
        if sample_rows > 0:
            sample_path = out_dir / f"pitch_sample_{min_season}_{max_season}.csv"
            pitch.head(sample_rows).to_pandas().to_csv(sample_path, index=False)

    hitters = build_hitters(pitch)
    pitchers = build_pitchers(pitch)
    pitch_types = build_pitch_types(pitch)
    team_hitting = build_team_hitting(pitch)
    team_pitching = build_team_pitching(pitch)

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
    )
    write_csv(hitter_pct, "hitter_pctiles.csv", out_dir)

    pitcher_pct = add_percentiles(
        pitchers,
        group_cols=["season", "level_id"],
        value_cols=[
            "std.ZQ",
            "std.DMG",
            "std.NRV",
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
    )
    write_csv(pitcher_pct, "pitcher_pctiles.csv", out_dir)

    pitch_types_pct = add_percentiles(
        pitch_types,
        group_cols=["season", "level_id", "pitch_tag"],
        value_cols=[
            "pct",
            "std.ZQ",
            "std.DMG",
            "std.NRV",
            "velo",
            "max_velo",
            "vaa",
            "haa",
            "ivb",
            "hb",
            "SwStr",
            "Ball_pct",
            "Z_Contact",
            "Chase",
            "CSW",
        ],
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
    parser.add_argument("--sample-rows", type=int, default=0, help="Write a pitch-level sample CSV")
    args = parser.parse_args()
    if args.sample_rows > 0:
        os.environ["POOBAH_SAMPLE_ROWS"] = str(args.sample_rows)
    main(min_season=args.min_season, max_season=args.max_season, out_dir=args.out_dir, level_ids=args.level_ids)
