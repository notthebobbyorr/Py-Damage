# -*- coding: utf-8 -*-
"""
Data Aggregation Script

Reads pitch-level parquet data (output from data_pull.py) and generates
aggregated CSV files for the Streamlit app.
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path
import unicodedata
from typing import Iterable
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
try:
    import psutil
except ImportError:  # optional dependency
    psutil = None

_REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = _REPO_DIR / "data" / "output"
RAW_DIR = _REPO_DIR / "data" / "raw"
OUT_DIR = DATA_DIR

STUFF_SCALE_MEAN = 50.0
STUFF_SCALE_STD = 10.0
POSITION_COUNT_COLS = ["UT", "C", "X1B", "X2B", "X3B", "SS", "OF", "P", "NA"]
POSITION_BINARY_MIN_COUNT = 20

ALL_STAR_DATES = {
    2025: "2025-07-15",
    2024: "2024-07-16",
    2023: "2023-07-11",
    2022: "2022-07-19",
    2021: "2021-07-13",
    2020: None,
    2019: "2019-07-09",
    2018: "2018-07-17",
    2017: "2017-07-11",
    2016: "2016-07-12",
    2015: "2015-07-14",
}


GAME_TYPE_GROUP_MAP = {
    "S": "Spring Training",
    "F": "Postseason",
    "D": "Postseason",
    "L": "Postseason",
    "W": "Postseason",
}


def _add_game_type_group(df: pl.DataFrame) -> pl.DataFrame:
    """Derive game_type_group from game_type. Missing/unknown game types → 'Regular Season'."""
    if "game_type" not in df.columns:
        return df.with_columns(pl.lit("Regular Season").alias("game_type_group"))
    return df.with_columns(
        pl.when(pl.col("game_type") == "S")
        .then(pl.lit("Spring Training"))
        .when(pl.col("game_type").is_in(["F", "D", "L", "W"]))
        .then(pl.lit("Postseason"))
        .otherwise(pl.lit("Regular Season"))
        .alias("game_type_group")
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


def _strip_accents(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _normalize_player_names(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    cols = [col for col in ["hitter_name", "name"] if col in df.columns]
    if not cols:
        return df
    return df.with_columns(
        [pl.col(col).map_elements(_strip_accents, return_dtype=pl.Utf8).alias(col) for col in cols]
    )


def _normalize_age_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    cols = [col for col in ["batter_age", "pitcher_age"] if col in df.columns]
    if not cols:
        return df
    return df.with_columns(
        [
            pl.col(col)
            .cast(pl.Float64, strict=False)
            .round(0)
            .cast(pl.Int64, strict=False)
            .alias(col)
            for col in cols
        ]
    )


def _optional_age_expr(df: pl.DataFrame, source_col: str) -> pl.Expr:
    if source_col not in df.columns:
        return pl.lit(None).cast(pl.Float64).alias("baseball_age")
    age_col = pl.col(source_col).cast(pl.Float64, strict=False)
    return (
        age_col.filter(age_col.is_not_null() & ~age_col.is_nan())
        .first()
        .alias("baseball_age")
    )


def _starter_role_expr(df: pl.DataFrame) -> pl.Expr:
    role_col = None
    if "pitcher_role" in df.columns:
        role_col = "pitcher_role"
    elif "role" in df.columns:
        role_col = "role"
    if role_col is None:
        return pl.lit(False)
    return (
        pl.col(role_col).cast(pl.Utf8).str.strip_chars().str.to_uppercase() == "SP"
    )


def _game_date_expr(df: pl.DataFrame) -> pl.Expr | None:
    if "game_date" not in df.columns:
        return None
    dtype = df.schema.get("game_date")
    if dtype == pl.Date:
        return pl.col("game_date")
    if dtype == pl.Datetime:
        return pl.col("game_date").cast(pl.Date)
    return pl.col("game_date").str.strptime(pl.Date, strict=False)


def _with_split(df: pl.DataFrame, split_expr: pl.Expr | None) -> pl.DataFrame | None:
    if split_expr is None:
        return None
    return df.with_columns(split_expr.alias("__split"))


def _split_vs_lr(df: pl.DataFrame, source_col: str) -> pl.Expr | None:
    if source_col not in df.columns:
        return None
    hand = pl.col(source_col).cast(pl.Utf8).str.to_uppercase()
    return (
        pl.when(hand == "L")
        .then(pl.lit("vs L"))
        .when(hand == "R")
        .then(pl.lit("vs R"))
        .otherwise(None)
    )


def _split_home_away(df: pl.DataFrame, hitter: bool) -> pl.Expr | None:
    if "home_team" in df.columns and "away_team" in df.columns:
        home_team = (
            pl.col("home_team").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
        )
        away_team = (
            pl.col("away_team").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
        )
        if hitter and "hitting_code" in df.columns:
            team = (
                pl.col("hitting_code")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_uppercase()
            )
            home_cond = team == home_team
            away_cond = team == away_team
        elif (not hitter) and "pitching_code" in df.columns:
            team = (
                pl.col("pitching_code")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_uppercase()
            )
            home_cond = team == home_team
            away_cond = team == away_team
        else:
            home_cond = away_cond = None
    else:
        home_cond = away_cond = None

    if home_cond is None or away_cond is None:
        if "inning_topbot" not in df.columns:
            return None
        topbot = (
            pl.col("inning_topbot").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
        )
        if hitter:
            home_cond = topbot.is_in(["bottom", "bot", "b"])
            away_cond = topbot.is_in(["top", "t"])
        else:
            home_cond = topbot.is_in(["top", "t"])
            away_cond = topbot.is_in(["bottom", "bot", "b"])
    return (
        pl.when(home_cond)
        .then(pl.lit("Home"))
        .when(away_cond)
        .then(pl.lit("Away"))
        .otherwise(None)
    )


def _split_month(df: pl.DataFrame) -> pl.Expr | None:
    month_expr = None
    if "Month" in df.columns:
        month_expr = pl.col("Month").cast(pl.Int64)
    else:
        date_expr = _game_date_expr(df)
        if date_expr is None:
            return None
        month_expr = date_expr.dt.month()

    return (
        pl.when(month_expr.is_in([3, 4]))
        .then(pl.lit("March/April"))
        .when(month_expr.is_in([9, 10]))
        .then(pl.lit("September/October"))
        .when(month_expr == 5)
        .then(pl.lit("May"))
        .when(month_expr == 6)
        .then(pl.lit("June"))
        .when(month_expr == 7)
        .then(pl.lit("July"))
        .when(month_expr == 8)
        .then(pl.lit("August"))
        .when(month_expr == 1)
        .then(pl.lit("January"))
        .when(month_expr == 2)
        .then(pl.lit("February"))
        .when(month_expr == 11)
        .then(pl.lit("November"))
        .when(month_expr == 12)
        .then(pl.lit("December"))
        .otherwise(None)
    )


def _split_half(df: pl.DataFrame) -> pl.DataFrame | None:
    if "season" not in df.columns or "game_date" not in df.columns:
        return None
    date_expr = _game_date_expr(df)
    if date_expr is None:
        return None
    asg_df = pl.DataFrame(
        {
            "season": list(ALL_STAR_DATES.keys()),
            "asg_date": list(ALL_STAR_DATES.values()),
        }
    ).with_columns(pl.col("asg_date").str.strptime(pl.Date, strict=False))
    split_df = df.with_columns(date_expr.alias("__game_date"))
    split_df = split_df.join(asg_df, on="season", how="left")
    split_df = split_df.with_columns(
        pl.when(
            pl.col("asg_date").is_not_null()
            & (pl.col("__game_date") < pl.col("asg_date"))
        )
        .then(pl.lit("1st Half"))
        .when(
            pl.col("asg_date").is_not_null()
            & (pl.col("__game_date") > pl.col("asg_date"))
        )
        .then(pl.lit("2nd Half"))
        .otherwise(None)
        .alias("__split")
    )
    return split_df.drop(["__game_date", "asg_date"])


def _build_split_frames(
    df: pl.DataFrame,
    build_fn,
    split_type: str,
    split_df_fn,
) -> list[pl.DataFrame]:
    if df.is_empty():
        return []
    split_df = split_df_fn(df)
    if split_df is None:
        return []
    labels = (
        split_df.select(pl.col("__split").drop_nulls().unique()).to_series().to_list()
    )
    frames: list[pl.DataFrame] = []
    for label in labels:
        subset = split_df.filter(pl.col("__split") == label)
        built = build_fn(subset)
        if built.is_empty():
            continue
        built = built.with_columns(
            pl.lit(split_type).alias("split_type"),
            pl.lit(label).alias("split"),
        )
        frames.append(built)
    return frames


def build_hitter_splits(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    split_specs = [
        ("vs L/R", lambda x: _with_split(x, _split_vs_lr(x, "throws"))),
        ("Home/Away", lambda x: _with_split(x, _split_home_away(x, hitter=True))),
        ("Monthly", lambda x: _with_split(x, _split_month(x))),
        ("1st Half/2nd Half", _split_half),
    ]
    frames: list[pl.DataFrame] = []
    for split_type, split_fn in split_specs:
        frames.extend(_build_split_frames(df, build_hitters, split_type, split_fn))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def build_pitching_splits(
    df: pl.DataFrame,
    stuff_percentiles: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if df.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    split_specs = [
        ("vs L/R", lambda x: _with_split(x, _split_vs_lr(x, "stands"))),
        ("Home/Away", lambda x: _with_split(x, _split_home_away(x, hitter=False))),
        ("Monthly", lambda x: _with_split(x, _split_month(x))),
        ("1st Half/2nd Half", _split_half),
    ]
    pitcher_frames: list[pl.DataFrame] = []
    pitch_type_frames: list[pl.DataFrame] = []
    for split_type, split_fn in split_specs:
        split_df = split_fn(df)
        if split_df is None:
            continue
        labels = (
            split_df.select(pl.col("__split").drop_nulls().unique())
            .to_series()
            .to_list()
        )
        for label in labels:
            subset = split_df.filter(pl.col("__split") == label)
            pitch_types_split = build_pitch_types(subset)
            pitch_types_split = apply_stuff_grade(pitch_types_split, stuff_percentiles)
            pitchers_split = build_pitchers(subset)
            if not pitch_types_split.is_empty():
                pitcher_stuff = (
                    pitch_types_split.filter(pl.col("stuff").is_not_null())
                    .group_by(["pitcher_mlbid", "season", "level_id", "game_type_group"])
                    .agg(
                        (pl.col("stuff") * pl.col("pitches")).sum()
                        / pl.col("pitches").sum()
                    )
                    .rename({"stuff": "stuff_grade"})
                )
                pitchers_split = (
                    pitchers_split.join(
                        pitcher_stuff,
                        on=["pitcher_mlbid", "season", "level_id", "game_type_group"],
                        how="left",
                    )
                    .with_columns(pl.col("stuff_grade").alias("stuff"))
                    .drop("stuff_grade")
                )
            if not pitchers_split.is_empty():
                pitchers_split = pitchers_split.with_columns(
                    pl.lit(split_type).alias("split_type"),
                    pl.lit(label).alias("split"),
                )
                pitcher_frames.append(pitchers_split)
            if not pitch_types_split.is_empty():
                pitch_types_split = pitch_types_split.with_columns(
                    pl.lit(split_type).alias("split_type"),
                    pl.lit(label).alias("split"),
                )
                pitch_type_frames.append(pitch_types_split)

    pitcher_splits = (
        pl.concat(pitcher_frames, how="diagonal") if pitcher_frames else pl.DataFrame()
    )
    pitch_types_splits = (
        pl.concat(pitch_type_frames, how="diagonal")
        if pitch_type_frames
        else pl.DataFrame()
    )
    return pitcher_splits, pitch_types_splits


def build_hitters(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    hitter_age_expr = _optional_age_expr(df, "batter_age")
    df = df.with_columns(
        pl.col("batter_position")
        .map_elements(_pos_label, return_dtype=pl.Utf8)
        .alias("position")
    )

    hitters = (
        df.group_by(["batter_mlbid", "hitter_name", "level_id", "season", "game_type_group"])
        .agg(
            [
                hitter_age_expr,
                pl.len().alias("pitches"),
                pl.n_unique("pa_id").alias("PA"),
                pl.sum("bbe").alias("bbe"),
                pl.quantile("exit_velo", 0.9).alias("EV90th"),
                pl.max("exit_velo").alias("max_EV"),
                (pl.col("is_in_play") == True).sum().alias("EV90th_n"),
                (pl.col("is_in_play") == True).sum().alias("max_EV_n"),
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
                    (pl.col("launch_angle") >= 20)
                    & (pl.col("spray_angle_adj") < -15)
                    & (pl.col("is_in_play") == True)
                )
                .sum()
                .alias("pull_FB_pct_num"),
                (pl.col("is_in_play") == True).sum().alias("pull_FB_pct_den"),
                (
                    100
                    * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                    / (pl.col("is_inzone_pi") == False).sum()
                ).alias("chase"),
                ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False))
                .sum()
                .alias("chase_num"),
                (pl.col("is_inzone_pi") == False).sum().alias("chase_den"),
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
                    (pl.col("whiff") != 1)
                    & (pl.col("swing") == 1)
                    & (pl.col("is_inzone_pi") == True)
                )
                .sum()
                .alias("z_con_num"),
                ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1))
                .sum()
                .alias("z_con_den"),
                (
                    100
                    * (
                        (pl.col("whiff") == 1) & (pl.col("pi_pitch_group") != "FA")
                    ).sum()
                    / (
                        (pl.col("swing") == 1) & (pl.col("pi_pitch_group") != "FA")
                    ).sum()
                ).alias("secondary_whiff_pct"),
                ((pl.col("whiff") == 1) & (pl.col("pi_pitch_group") != "FA"))
                .sum()
                .alias("secondary_whiff_pct_num"),
                ((pl.col("swing") == 1) & (pl.col("pi_pitch_group") != "FA"))
                .sum()
                .alias("secondary_whiff_pct_den"),
                (
                    100
                    * ((pl.col("whiff") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                    / ((pl.col("swing") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                ).alias("whiffs_vs_95"),
                ((pl.col("whiff") == 1) & (pl.col("pitch_velo") >= 95))
                .sum()
                .alias("whiffs_vs_95_num"),
                ((pl.col("swing") == 1) & (pl.col("pitch_velo") >= 95))
                .sum()
                .alias("whiffs_vs_95_den"),
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
                    (pl.col("is_in_play") == True)
                    & (pl.col("damage_pred").is_not_null())
                    & (pl.col("exit_velo") >= pl.col("damage_pred"))
                    & (pl.col("launch_angle") > 0)
                    & (pl.col("spray_angle_adj") >= -50)
                    & (pl.col("spray_angle_adj") <= 50)
                )
                .sum()
                .alias("damage_rate_num"),
                (pl.col("is_in_play") == True).sum().alias("damage_rate_den"),
                (
                    100
                    * ((pl.col("decision_value") > 0) & (pl.col("swing") == 0)).sum()
                    / (pl.col("decision_value") > 0).sum()
                ).alias("selection_skill"),
                ((pl.col("decision_value") > 0) & (pl.col("swing") == 0))
                .sum()
                .alias("selection_skill_num"),
                (pl.col("decision_value") > 0).sum().alias("selection_skill_den"),
                (
                    100
                    * ((pl.col("decision_value") < 0) & (pl.col("swing") == 0)).sum()
                    / (pl.col("swing") == 0).sum()
                ).alias("hittable_pitches_taken"),
                ((pl.col("decision_value") < 0) & (pl.col("swing") == 0))
                .sum()
                .alias("hittable_pitches_taken_num"),
                (pl.col("swing") == 0).sum().alias("hittable_pitches_taken_den"),
                (
                    pl.when(pl.col("swing") == 1).then(pl.col("pred_whiff_loc")).mean()
                ).alias("pred_whiff_loc_mean"),
                (pl.col("swing") == 1).sum().alias("pred_whiff_loc_mean_n"),
                (pl.col("whiff") == 1).sum().alias("whiff_rate_num"),
                (pl.col("swing") == 1).sum().alias("whiff_rate_den"),
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
                ((pl.col("launch_angle") < 0) & (pl.col("is_in_play") == True))
                .sum()
                .alias("LA_lte_0_num"),
                (pl.col("is_in_play") == True).sum().alias("LA_lte_0_den"),
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
                    (pl.col("launch_angle") >= 0)
                    & (pl.col("launch_angle") <= 20)
                    & (pl.col("is_in_play") == True)
                )
                .sum()
                .alias("LD_pct_num"),
                (pl.col("is_in_play") == True).sum().alias("LD_pct_den"),
                (
                    100
                    * (
                        (pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True)
                    ).sum()
                    / (pl.col("is_in_play") == True).sum()
                ).alias("LA_gte_20"),
                ((pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True))
                .sum()
                .alias("LA_gte_20_num"),
                (pl.col("is_in_play") == True).sum().alias("LA_gte_20_den"),
                pl.mean("bat_speed").alias("bat_speed"),
                pl.col("bat_speed").is_not_null().sum().alias("bat_speed_n"),
                pl.mean("swing_length").alias("swing_length"),
                pl.col("swing_length").is_not_null().sum().alias("swing_length_n"),
                pl.mean("attack_angle").alias("attack_angle"),
                pl.col("attack_angle").is_not_null().sum().alias("attack_angle_n"),
                pl.mean("swing_path_tilt").alias("swing_path_tilt"),
                pl.col("swing_path_tilt")
                .is_not_null()
                .sum()
                .alias("swing_path_tilt_n"),
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
                (pl.col("pitch_outcome") == "HR").sum().alias("HR"),
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
            on="position",
        )
        .fill_null(0)
    )

    hitters = hitters.join(
        pos_counts,
        on=["batter_mlbid", "hitter_name", "level_id", "season"],
        how="left",
    )

    for col in POSITION_COUNT_COLS:
        if col not in hitters.columns:
            hitters = hitters.with_columns(pl.lit(0).alias(col))
    hitters = hitters.with_columns(
        [
            pl.when(pl.col(col) >= POSITION_BINARY_MIN_COUNT)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"is_{col}")
            for col in POSITION_COUNT_COLS
            if col != "NA"
        ]
    )

    return hitters


def build_pitchers(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    pitcher_age_expr = _optional_age_expr(df, "pitcher_age")
    starter_role = _starter_role_expr(df)
    gs_expr = (
        pl.col("game_pk").filter(starter_role).n_unique().alias("GS")
        if "game_pk" in df.columns
        else pl.lit(0).alias("GS")
    )

    fastball_tags = ["FA", "SI", "HC", "SP"]
    fb_primary_tag = (
        df.filter(pl.col("pitch_tag").is_in(fastball_tags))
        .group_by(["pitcher_mlbid", "level_id", "season", "game_type_group", "pitch_tag"])
        .agg(pl.len().alias("pitch_count"))
        .sort(
            ["pitcher_mlbid", "level_id", "season", "game_type_group", "pitch_count"],
            descending=[False, False, False, False, True],
        )
        .group_by(["pitcher_mlbid", "level_id", "season", "game_type_group"])
        .agg(pl.first("pitch_tag").alias("primary_fb_tag"))
    )
    fb_vaa = (
        df.join(
            fb_primary_tag,
            on=["pitcher_mlbid", "level_id", "season", "game_type_group"],
            how="left",
        )
        .filter(pl.col("pitch_tag") == pl.col("primary_fb_tag"))
        .group_by(["pitcher_mlbid", "level_id", "season", "game_type_group"])
        .agg(pl.mean("vaa").alias("fastball_vaa_override"))
    )

    pitchers = df.group_by(
        ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand", "game_type_group"]
    ).agg(
        [
            pitcher_age_expr,
            pl.len().alias("pitches"),
            pl.n_unique("pa_id").alias("TBF"),
            gs_expr,
            (pl.sum("outs_recorded") / 3).round(1).alias("IP"),
            (pl.n_unique("pa_id") / pl.n_unique("game_pk")).alias("TBF_per_G"),
            (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
            (pl.col("whiff") == 1).sum().alias("SwStr_num"),
            pl.len().alias("SwStr_den"),
            ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
            (pl.col("is_ball") == True).sum().alias("Ball_pct_num"),
            pl.len().alias("Ball_pct_den"),
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
                (pl.col("whiff") != 1)
                & (pl.col("swing") == 1)
                & (pl.col("is_inzone_pi") == True)
            )
            .sum()
            .alias("Z_Contact_num"),
            ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1))
            .sum()
            .alias("Z_Contact_den"),
            ((pl.col("is_inzone_pi") == True).sum() / pl.len()).mul(100).alias("Zone"),
            (pl.col("is_inzone_pi") == True).sum().alias("Zone_num"),
            pl.len().alias("Zone_den"),
            (
                100
                * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                / (pl.col("is_inzone_pi") == False).sum()
            ).alias("Chase"),
            ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False))
            .sum()
            .alias("Chase_num"),
            (pl.col("is_inzone_pi") == False).sum().alias("Chase_den"),
            (
                100
                * (
                    ((pl.col("whiff") == 1)).sum()
                    + ((pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)).sum()
                )
                / pl.len()
            ).alias("CSW"),
            (
                ((pl.col("whiff") == 1)).sum()
                + ((pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)).sum()
            ).alias("CSW_num"),
            pl.len().alias("CSW_den"),
            (100 * pl.col("pred_whiff_base").mean()).alias("pWhiff"),
            pl.col("pred_whiff_base").is_not_null().sum().alias("pWhiff_n"),
            pl.mean("loc_adj_vaa").alias("loc_adj_vaa"),
            pl.col("loc_adj_vaa").is_not_null().sum().alias("loc_adj_vaa_n"),
            (100 * (pl.col("pitch_group") == "FA").sum() / pl.len()).alias("FA_pct"),
            (pl.col("pitch_group") == "FA").sum().alias("FA_pct_num"),
            pl.len().alias("FA_pct_den"),
            (pl.when(pl.col("pitch_group") == "BR").then(pl.col("rpm")).mean()).alias(
                "BB_rpm"
            ),
            ((pl.col("pitch_group") == "BR") & (pl.col("rpm").is_not_null()))
            .sum()
            .alias("BB_rpm_n"),
            (
                pl.when(pl.col("pitch_group") == "FA")
                .then(pl.col("spin_efficiency") * 100)
                .mean()
            ).alias("FA_spin_eff"),
            (
                (pl.col("pitch_group") == "FA")
                & (pl.col("spin_efficiency").is_not_null())
            )
            .sum()
            .alias("FA_spin_eff_n"),
            (
                100
                * ((pl.col("launch_angle") <= 0) & (pl.col("is_in_play") == True)).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_lte_0"),
            ((pl.col("launch_angle") <= 0) & (pl.col("is_in_play") == True))
            .sum()
            .alias("LA_lte_0_num"),
            (pl.col("is_in_play") == True).sum().alias("LA_lte_0_den"),
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
                (pl.col("launch_angle") >= 0)
                & (pl.col("launch_angle") <= 20)
                & (pl.col("is_in_play") == True)
            )
            .sum()
            .alias("LD_pct_num"),
            (pl.col("is_in_play") == True).sum().alias("LD_pct_den"),
            (
                100
                * (
                    (pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True)
                ).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_gte_20"),
            ((pl.col("launch_angle") >= 20) & (pl.col("is_in_play") == True))
            .sum()
            .alias("LA_gte_20_num"),
            (pl.col("is_in_play") == True).sum().alias("LA_gte_20_den"),
            pl.mean("primary_velo").alias("fastball_velo"),
            pl.col("primary_velo").is_not_null().sum().alias("fastball_velo_n"),
            pl.max("pitch_velo").alias("max_velo"),
            pl.col("pitch_velo").is_not_null().sum().alias("max_velo_n"),
            pl.mean("primary_vaa").alias("fastball_vaa"),
            pl.col("primary_vaa").is_not_null().sum().alias("fastball_vaa_n"),
            pl.mean("stuff_raw").alias("stuff_raw"),
            pl.col("stuff_raw").is_not_null().sum().alias("stuff_raw_n"),
            pl.mean("release_z").alias("rel_z"),
            pl.col("release_z").is_not_null().sum().alias("rel_z_n"),
            pl.mean("release_x").alias("rel_x"),
            pl.col("release_x").is_not_null().sum().alias("rel_x_n"),
            pl.mean("ext").alias("ext"),
            pl.col("ext").is_not_null().sum().alias("ext_n"),
            pl.mean("arm_angle").alias("arm_angle"),
            pl.col("arm_angle").is_not_null().sum().alias("arm_angle_n"),
            pl.col("primary_tag")
            .filter(pl.col("primary_tag").is_not_null())
            .unique()
            .implode()
            .list.join(", ")
            .alias("primary_pitches"),
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
            (pl.col("pitch_outcome") == "HR").sum().alias("HR"),
        ]
    )
    pitchers = pitchers.join(
        fb_vaa, on=["pitcher_mlbid", "level_id", "season", "game_type_group"], how="left"
    ).with_columns(
        pl.coalesce(["fastball_vaa_override", "fastball_vaa"]).alias("fastball_vaa")
    ).drop("fastball_vaa_override")
    return pitchers


def build_pitch_types(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    pitcher_age_expr = _optional_age_expr(df, "pitcher_age")
    df = _tag_pitch(df)
    df = df.with_columns(
        pl.len()
        .over(
            [
                "name",
                "level_id",
                "pitcher_mlbid",
                "pitcher_hand",
                "season",
                "game_type_group",
            ]
        )
        .alias("total_pitches")
    )
    pitch_types = df.group_by(
        [
            "name",
            "level_id",
            "pitcher_mlbid",
            "pitcher_hand",
            "season",
            "game_type_group",
            "pitch_tag",
        ]
    ).agg(
        [
            pitcher_age_expr,
            pl.len().alias("pitches"),
            (pl.len() / pl.first("total_pitches") * 100).alias("pct"),
            pl.mean("stuff_raw").alias("stuff_raw"),
            pl.col("stuff_raw").is_not_null().sum().alias("stuff_raw_n"),
            pl.mean("pitch_velo").alias("velo"),
            pl.col("pitch_velo").is_not_null().sum().alias("velo_n"),
            pl.max("pitch_velo").alias("max_velo"),
            pl.col("pitch_velo").is_not_null().sum().alias("max_velo_n"),
            pl.mean("vaa").alias("vaa"),
            pl.col("vaa").is_not_null().sum().alias("vaa_n"),
            pl.mean("haa").alias("haa"),
            pl.col("haa").is_not_null().sum().alias("haa_n"),
            pl.mean("vbreak").alias("vbreak"),
            pl.col("vbreak").is_not_null().sum().alias("vbreak_n"),
            pl.mean("hbreak").alias("hbreak"),
            pl.col("hbreak").is_not_null().sum().alias("hbreak_n"),
            pl.mean("loc_adj_vaa").alias("loc_adj_vaa"),
            pl.col("loc_adj_vaa").is_not_null().sum().alias("loc_adj_vaa_n"),
            pl.mean("rpm").alias("rpm"),
            pl.col("rpm").is_not_null().sum().alias("rpm_n"),
            pl.mean("axis").alias("axis"),
            pl.col("axis").is_not_null().sum().alias("axis_n"),
            (pl.mean("spin_efficiency") * 100).alias("spin_efficiency"),
            pl.col("spin_efficiency").is_not_null().sum().alias("spin_efficiency_n"),
            pl.col("primary_tag")
            .filter(pl.col("primary_tag").is_not_null())
            .unique()
            .implode()
            .list.join(", ")
            .alias("primary_pitches"),
            pl.mean("primary_loc_adj_vaa").alias("primary_loc_adj_vaa"),
            pl.col("primary_loc_adj_vaa")
            .is_not_null()
            .sum()
            .alias("primary_loc_adj_vaa_n"),
            pl.mean("primary_velo").alias("primary_velo"),
            pl.col("primary_velo").is_not_null().sum().alias("primary_velo_n"),
            pl.mean("primary_rpm").alias("primary_rpm"),
            pl.col("primary_rpm").is_not_null().sum().alias("primary_rpm_n"),
            pl.mean("primary_axis").alias("primary_axis"),
            pl.col("primary_axis").is_not_null().sum().alias("primary_axis_n"),
            pl.mean("primary_hbreak").alias("primary_hbreak"),
            pl.col("primary_hbreak").is_not_null().sum().alias("primary_hbreak_n"),
            pl.mean("primary_vbreak").alias("primary_vbreak"),
            pl.col("primary_vbreak").is_not_null().sum().alias("primary_vbreak_n"),
            pl.mean("primary_z_release").alias("primary_z_release"),
            pl.col("primary_z_release")
            .is_not_null()
            .sum()
            .alias("primary_z_release_n"),
            pl.mean("primary_x_release").alias("primary_x_release"),
            pl.col("primary_x_release")
            .is_not_null()
            .sum()
            .alias("primary_x_release_n"),
            (
                100
                * ((pl.col("launch_angle") <= 0) & (pl.col("is_in_play") == True)).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_lte_0"),
            ((pl.col("launch_angle") <= 0) & (pl.col("is_in_play") == True))
            .sum()
            .alias("LA_lte_0_num"),
            (pl.col("is_in_play") == True).sum().alias("LA_lte_0_den"),
            (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
            (pl.col("whiff") == 1).sum().alias("SwStr_num"),
            pl.len().alias("SwStr_den"),
            ((pl.col("is_inzone_pi") == True).sum() / pl.len()).mul(100).alias("Zone"),
            (pl.col("is_inzone_pi") == True).sum().alias("Zone_num"),
            pl.len().alias("Zone_den"),
            ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
            (pl.col("is_ball") == True).sum().alias("Ball_pct_num"),
            pl.len().alias("Ball_pct_den"),
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
                (pl.col("whiff") != 1)
                & (pl.col("swing") == 1)
                & (pl.col("is_inzone_pi") == True)
            )
            .sum()
            .alias("Z_Contact_num"),
            ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1))
            .sum()
            .alias("Z_Contact_den"),
            (
                100
                * ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum()
                / (pl.col("is_inzone_pi") == False).sum()
            ).alias("Chase"),
            ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False))
            .sum()
            .alias("Chase_num"),
            (pl.col("is_inzone_pi") == False).sum().alias("Chase_den"),
            (
                100
                * (
                    ((pl.col("whiff") == 1)).sum()
                    + ((pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)).sum()
                )
                / pl.len()
            ).alias("CSW"),
            (
                ((pl.col("whiff") == 1)).sum()
                + ((pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)).sum()
            ).alias("CSW_num"),
            pl.len().alias("CSW_den"),
            (
                100
                * pl.when(pl.col("swing") == 1).then(pl.col("pred_whiff_base")).mean()
            ).alias("pred_whiff_pct"),
            ((pl.col("swing") == 1) & (pl.col("pred_whiff_base").is_not_null()))
            .sum()
            .alias("pred_whiff_pct_n"),
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
            (pl.col("pitch_outcome") == "HR").sum().alias("HR"),
            pl.mean("release_z").alias("rel_z"),
            pl.col("release_z").is_not_null().sum().alias("rel_z_n"),
            pl.mean("release_x").alias("rel_x"),
            pl.col("release_x").is_not_null().sum().alias("rel_x_n"),
            pl.mean("ext").alias("ext"),
            pl.col("ext").is_not_null().sum().alias("ext_n"),
            pl.mean("arm_angle").alias("arm_angle"),
            pl.col("arm_angle").is_not_null().sum().alias("arm_angle_n"),
        ]
    )
    if "LA_lte_0" not in pitch_types.columns:
        pitch_types = pitch_types.with_columns(
            [
                pl.lit(None).alias("LA_lte_0"),
                pl.lit(0).alias("LA_lte_0_num"),
                pl.lit(0).alias("LA_lte_0_den"),
            ]
        )
    return pitch_types


def build_league_pitch_types(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _tag_pitch(df)
    league_pitch_types = df.group_by(
        ["season", "level_id", "game_type_group", "pitcher_hand", "pitch_tag"]
    ).agg(
        [
            pl.len().alias("pitches"),
            pl.mean("pitch_velo").alias("velo"),
            pl.mean("vaa").alias("vaa"),
            pl.mean("haa").alias("haa"),
            pl.mean("vbreak").alias("vbreak"),
            pl.mean("hbreak").alias("hbreak"),
            (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
            (
                100
                * ((pl.col("launch_angle") <= 0) & (pl.col("is_in_play") == True)).sum()
                / (pl.col("is_in_play") == True).sum()
            ).alias("LA_lte_0"),
            (
                100
                * (
                    (pl.col("whiff") != 1)
                    & (pl.col("swing") == 1)
                    & (pl.col("is_inzone_pi") == True)
                ).sum()
                / ((pl.col("is_inzone_pi") == True) & (pl.col("swing") == 1)).sum()
            ).alias("Z_Contact"),
            ((pl.col("is_inzone_pi") == True).sum() / pl.len()).mul(100).alias("Zone"),
            ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
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
        ]
    )
    league_pitch_types = league_pitch_types.with_columns(
        (
            100
            * pl.col("pitches")
            / pl.col("pitches").sum().over(["season", "level_id", "pitcher_hand"])
        ).alias("pct"),
        pl.col("pitcher_hand").alias("throws"),
    )
    return league_pitch_types


_TEAM_CODE_ALIASES: dict[str, str] = {
    "AZ": "ARI",
}


def _normalize_team_codes(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Remap historical team code aliases to canonical codes."""
    if col not in df.columns:
        return df
    old_codes = list(_TEAM_CODE_ALIASES.keys())
    new_codes = list(_TEAM_CODE_ALIASES.values())
    return df.with_columns(
        pl.col(col).replace(old_codes, new_codes).alias(col)
    )


def build_team_hitting(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _normalize_team_codes(df, "hitting_code")
    team = (
        df.filter(
            pl.col("hitting_code").is_not_null()
            & ~pl.col("hitting_code").str.contains(r"^\d+$")
        )
        .group_by(["hitting_code", "level_id", "season", "game_type_group"])
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
                    * ((pl.col("whiff") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                    / ((pl.col("swing") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                ).alias("whiffs_vs_95"),
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


def build_league_hitting(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    league = (
        df.group_by(["level_id", "season", "game_type_group"])
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
                    * ((pl.col("whiff") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                    / ((pl.col("swing") == 1) & (pl.col("pitch_velo") >= 95)).sum()
                ).alias("whiffs_vs_95"),
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
    return league


def build_team_pitching(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _normalize_team_codes(df, "pitching_code")
    team = (
        df.filter(
            pl.col("pitching_code").is_not_null()
            & ~pl.col("pitching_code").str.contains(r"^\d+$")
        )
        .group_by(["pitching_code", "level_id", "season", "game_type_group"])
        .agg(
            [
                pl.n_unique("pa_id").alias("TBF"),
                (pl.sum("outs_recorded") / 3).round(1).alias("IP"),
                pl.sum("bbe").alias("bbe"),
                pl.len().alias("pitches"),
                pl.mean("stuff_raw").alias("stuff_raw"),
                pl.mean("loc_adj_vaa").alias("loc_adj_vaa"),
                (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
                ((pl.col("is_inzone_pi") == True).sum() / pl.len())
                .mul(100)
                .alias("Zone"),
                ((pl.col("is_ball") == True).sum() / pl.len())
                .mul(100)
                .alias("Ball_pct"),
                (100 * (pl.col("pitch_group") == "FA").sum() / pl.len()).alias(
                    "FA_pct"
                ),
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
                        + (
                            (pl.col("pitch_outcome") == "S") & (pl.col("swing") == 0)
                        ).sum()
                    )
                    / pl.len()
                ).alias("CSW"),
                (100 * pl.col("pred_whiff_base").mean()).alias("pWhiff"),
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
                pl.mean("primary_velo").alias("fastball_velo"),
                pl.mean("primary_vaa").alias("fastball_vaa"),
            ]
        )
    )
    return team


def build_league_pitching(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    league = df.group_by(["level_id", "season", "game_type_group"]).agg(
        [
            pl.n_unique("pa_id").alias("TBF"),
            (pl.sum("outs_recorded") / 3).round(1).alias("IP"),
            pl.sum("bbe").alias("bbe"),
            pl.len().alias("pitches"),
            pl.mean("stuff_raw").alias("stuff_raw"),
            (pl.sum("whiff") / pl.len()).mul(100).alias("SwStr"),
            ((pl.col("is_ball") == True).sum() / pl.len()).mul(100).alias("Ball_pct"),
            (100 * (pl.col("pitch_group") == "FA").sum() / pl.len()).alias("FA_pct"),
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
    return league


def build_park_data(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    required = {
        "park_mlbid",
        "stands",
        "season",
        "level_id",
        "pitch_outcome",
        "home_team",
        "damage_pred",
        "exit_velo",
        "launch_angle",
        "spray_angle_adj",
        "is_in_play",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for park_data: {sorted(missing)}")
    damage_mask = (
        (pl.col("is_in_play") == True)
        & (pl.col("damage_pred").is_not_null())
        & (pl.col("exit_velo") >= pl.col("damage_pred"))
        & (pl.col("launch_angle") > 0)
        & (pl.col("spray_angle_adj") >= -50)
        & (pl.col("spray_angle_adj") <= 50)
    )
    bbe_mask = (
        (pl.col("bbe") == 1) if "bbe" in df.columns else (pl.col("is_in_play") == True)
    )
    hit_mask = pl.col("pitch_outcome").is_in(["1B", "2B", "3B", "HR"])
    xbh_mask = pl.col("pitch_outcome").is_in(["2B", "3B", "HR"])
    la_lte_0_mask = bbe_mask & (pl.col("launch_angle") <= 0)
    la_0_to_20_mask = bbe_mask & (pl.col("launch_angle") > 0) & (pl.col("launch_angle") < 20)
    la20_mask = bbe_mask & (pl.col("launch_angle") >= 20)
    pulled_fb_mask = la20_mask & (pl.col("spray_angle_adj") < -15)
    non_damage_mask = bbe_mask & (~damage_mask)
    non_damage_la20_mask = non_damage_mask & (pl.col("launch_angle") >= 20)
    hr_bbe = (pl.col("pitch_outcome") == "HR") & bbe_mask
    hr_damage = (pl.col("pitch_outcome") == "HR") & damage_mask
    hr_non_damage = (pl.col("pitch_outcome") == "HR") & non_damage_mask
    hr_non_damage_la20 = (pl.col("pitch_outcome") == "HR") & non_damage_la20_mask
    hr_la20 = hr_bbe & (pl.col("launch_angle") >= 20)
    hr_pulled_fb = hr_bbe & (pl.col("spray_angle_adj") < -15) & (pl.col("launch_angle") >= 20)
    xbh_damage = xbh_mask & damage_mask
    xbh_bbe = xbh_mask & bbe_mask
    hit_bbe = hit_mask & bbe_mask
    hits_la_lte_0 = hit_mask & la_lte_0_mask
    hits_la_0_to_20 = hit_mask & la_0_to_20_mask
    hits_la_gte_20 = hit_mask & la20_mask
    park = (
        df.filter(
            pl.col("park_mlbid").is_not_null()
            & pl.col("home_team").is_not_null()
        )
        .group_by(["park_mlbid", "home_team", "stands", "season", "level_id"])
        .agg(
            [
                damage_mask.sum().alias("damage_bbe"),
                non_damage_mask.sum().alias("non_damage_bbe"),
                non_damage_la20_mask.sum().alias("non_damage_la_gte_20_bbe"),
                hr_damage.sum().alias("hr_damage_bbe"),
                hr_non_damage.sum().alias("hr_non_damage_bbe"),
                hr_non_damage_la20.sum().alias("hr_non_damage_la_gte_20_bbe"),
                la_lte_0_mask.sum().alias("la_lte_0_bbe"),
                la_0_to_20_mask.sum().alias("la_0_to_20_bbe"),
                xbh_damage.sum().alias("xbh_damage_bbe"),
                la20_mask.sum().alias("la_gte_20_bbe"),
                pulled_fb_mask.sum().alias("pulled_fb_bbe"),
                hr_la20.sum().alias("hr_la_gte_20"),
                hr_pulled_fb.sum().alias("hr_pulled_fb"),
                hr_bbe.sum().alias("hr_bbe"),
                xbh_bbe.sum().alias("xbh_bbe"),
                hit_bbe.sum().alias("hits_bbe"),
                hits_la_lte_0.sum().alias("hits_la_lte_0"),
                hits_la_0_to_20.sum().alias("hits_la_0_to_20"),
                hits_la_gte_20.sum().alias("hits_la_gte_20"),
                bbe_mask.sum().alias("bbe_total"),
            ]
        )
        .filter(pl.col("damage_bbe") >= 100)
        .with_columns(
            pl.when(pl.col("damage_bbe") > 0)
            .then((pl.col("hr_damage_bbe") / pl.col("damage_bbe")) * 100)
            .otherwise(None)
            .alias("HR_per_damage_BBE_pct")
        )
        .with_columns(
            pl.when(pl.col("damage_bbe") > 0)
            .then((pl.col("xbh_damage_bbe") / pl.col("damage_bbe")) * 100)
            .otherwise(None)
            .alias("XBH_per_damage_BBE_pct"),
            pl.when(pl.col("la_gte_20_bbe") > 0)
            .then((pl.col("hr_la_gte_20") / pl.col("la_gte_20_bbe")) * 100)
            .otherwise(None)
            .alias("HR_per_LA_gte_20_pct"),
            pl.when(pl.col("pulled_fb_bbe") > 0)
            .then((pl.col("hr_pulled_fb") / pl.col("pulled_fb_bbe")) * 100)
            .otherwise(None)
            .alias("HR_per_pulled_FB_pct"),
            pl.when(pl.col("bbe_total") > 0)
            .then((pl.col("hr_bbe") / pl.col("bbe_total")) * 100)
            .otherwise(None)
            .alias("HR_per_BBE_pct"),
            pl.when(pl.col("non_damage_bbe") > 0)
            .then((pl.col("hr_non_damage_bbe") / pl.col("non_damage_bbe")) * 100)
            .otherwise(None)
            .alias("HR_per_non_damage_BBE_pct"),
            pl.when(pl.col("non_damage_la_gte_20_bbe") > 0)
            .then((pl.col("hr_non_damage_la_gte_20_bbe") / pl.col("non_damage_la_gte_20_bbe")) * 100)
            .otherwise(None)
            .alias("HR_per_non_damage_LA_gte_20_BBE_pct"),
            pl.when(pl.col("bbe_total") > 0)
            .then((pl.col("xbh_bbe") / pl.col("bbe_total")) * 100)
            .otherwise(None)
            .alias("XBH_per_BBE_pct"),
            pl.when(pl.col("la_lte_0_bbe") > 0)
            .then((pl.col("hits_la_lte_0") / pl.col("la_lte_0_bbe")) * 100)
            .otherwise(None)
            .alias("Hits_per_LA_lte_0_pct"),
            pl.when(pl.col("la_0_to_20_bbe") > 0)
            .then((pl.col("hits_la_0_to_20") / pl.col("la_0_to_20_bbe")) * 100)
            .otherwise(None)
            .alias("Hits_per_LA_0_to_20_pct"),
            pl.when(pl.col("la_gte_20_bbe") > 0)
            .then((pl.col("hits_la_gte_20") / pl.col("la_gte_20_bbe")) * 100)
            .otherwise(None)
            .alias("Hits_per_LA_gte_20_pct"),
            pl.when(pl.col("bbe_total") > 0)
            .then((pl.col("hits_bbe") / pl.col("bbe_total")) * 100)
            .otherwise(None)
            .alias("Hits_per_BBE_pct"),
        )
    )
    return park


def compute_stuff_percentiles(
    df: pl.DataFrame,
    raw_col: str = "stuff_raw",
    min_pitches: int = 50,
) -> pl.DataFrame:
    """Compute stuff grade percentile thresholds from MLB only (level_id=1)."""
    if df.is_empty() or raw_col not in df.columns:
        return pl.DataFrame()

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

    stats = pitcher_avgs.group_by(["season", "pitch_tag"]).agg(
        [
            pl.col("pitcher_stuff_avg").quantile(0.01).alias("stuff_p01"),
            pl.col("pitcher_stuff_avg").quantile(0.99).alias("stuff_p99"),
            pl.len().alias("n_pitchers"),
        ]
    )

    fallback_stats = pitcher_avgs.group_by(["pitch_tag"]).agg(
        [
            pl.col("pitcher_stuff_avg").quantile(0.01).alias("fallback_p01"),
            pl.col("pitcher_stuff_avg").quantile(0.99).alias("fallback_p99"),
        ]
    )

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
    """Apply stuff grades to aggregated data using precomputed percentiles."""
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
            .cast(pl.Int64, strict=False)
        )
        .otherwise(pl.lit(None))
    )
    return df.with_columns([grade_expr.alias(grade_col)]).drop(
        ["stuff_p01", "stuff_p99"]
    )


def add_percentiles(
    df: pl.DataFrame,
    group_cols: Iterable[str],
    value_cols: Iterable[str],
    filter_col: str | None = None,
    min_threshold: float | None = None,
    bins: int = 100,
) -> pl.DataFrame:
    """Add integer percentile columns for value_cols, grouped by group_cols."""
    if df.is_empty():
        return df
    df_pd = df.to_pandas()
    group_list = list(group_cols)
    value_list = list(value_cols)

    def _pctile_bins(series: pd.Series) -> pd.Series:
        series = series.copy()
        series = series.apply(lambda v: v if np.isscalar(v) else np.nan)
        series = pd.to_numeric(series, errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            return pd.Series([pd.NA] * len(series), index=series.index)
        try:
            pct = pd.qcut(
                non_null,
                q=bins,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            return pd.Series([pd.NA] * len(series), index=series.index)
        # Map to 1..N bins
        pct = pct + 1
        out = pd.Series([pd.NA] * len(series), index=series.index)
        out.loc[pct.index] = pct.astype("Int64")
        return out

    if filter_col and min_threshold is not None and filter_col in df_pd.columns:
        qualified = df_pd[df_pd[filter_col] >= min_threshold].copy()
        if qualified.empty:
            for col in value_list:
                df_pd[f"{col}_pctile"] = pd.NA
            return pl.from_pandas(df_pd)

        for col in value_list:
            pct = qualified.groupby(group_list, group_keys=False)[col].apply(
                _pctile_bins
            )
            df_pd[f"{col}_pctile"] = pd.NA
            df_pd.loc[pct.index, f"{col}_pctile"] = pct
    else:
        for col in value_list:
            pct = df_pd.groupby(group_list, group_keys=False)[col].apply(_pctile_bins)
            df_pd[f"{col}_pctile"] = pct

    return pl.from_pandas(df_pd)


def write_parquet(df: pl.DataFrame, name: str, out_dir: Path) -> None:
    path = out_dir / name
    start = perf_counter()
    try:
        df.write_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed writing {path}") from exc
    print(f"Wrote {len(df):,} rows to {path}")
    print(f"Write time: {perf_counter() - start:0.2f}s")


def _weighted_pct_merge(
    df: pl.DataFrame,
    *,
    keys: list[str],
    weight_col: str,
    value_cols: list[str],
) -> pl.DataFrame:
    agg_exprs: list[pl.Expr] = [pl.col(weight_col).sum().alias(weight_col)]
    for col in value_cols:
        agg_exprs.append(
            ((pl.col(col) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()).alias(col)
        )
    return df.group_by(keys).agg(agg_exprs)


def _apply_p_swstr_sources(
    pitchers: pl.DataFrame,
    pitch_types: pl.DataFrame,
    *,
    input_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pitcher_source_path = input_dir / "pitcher_p_swstr.parquet"
    pitch_type_source_path = input_dir / "pitch_types_p_swstr.parquet"

    if pitcher_source_path.exists():
        pitcher_source = pl.read_parquet(pitcher_source_path).rename({"pitcher_name": "name"})
        pitcher_keys = ["pitcher_mlbid", "season"]
        if "level_id" in pitcher_source.columns:
            pitcher_keys.append("level_id")
        pitcher_source = _weighted_pct_merge(
            pitcher_source,
            keys=pitcher_keys,
            weight_col="n",
            value_cols=["p_SwStr_pct", "p_SwStr_with_loc_pct"],
        ).rename({"n": "p_SwStr_n"})
        pitchers = pitchers.join(
            pitcher_source,
            on=pitcher_keys,
            how="left",
        )

    if pitch_type_source_path.exists():
        pitch_type_source = pl.read_parquet(pitch_type_source_path).rename({"pitcher_name": "name"})
        pitch_type_keys = ["pitcher_mlbid", "season", "pitch_tag"]
        if "level_id" in pitch_type_source.columns:
            pitch_type_keys.append("level_id")
        pitch_type_source = _weighted_pct_merge(
            pitch_type_source,
            keys=pitch_type_keys,
            weight_col="n",
            value_cols=["p_SwStr_pct", "p_SwStr_with_loc_pct"],
        ).rename({"n": "p_SwStr_n"})
        pitch_types = pitch_types.join(
            pitch_type_source,
            on=pitch_type_keys,
            how="left",
        )

    if "p_SwStr_pct" not in pitchers.columns:
        pitchers = pitchers.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("p_SwStr_pct"),
                pl.lit(None, dtype=pl.Float64).alias("p_SwStr_with_loc_pct"),
            ]
        )
    if "p_SwStr_pct" not in pitch_types.columns:
        pitch_types = pitch_types.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("p_SwStr_pct"),
                pl.lit(None, dtype=pl.Float64).alias("p_SwStr_with_loc_pct"),
            ]
        )

    return pitchers, pitch_types


def _apply_p_damage_sources(
    pitchers: pl.DataFrame,
    pitch_types: pl.DataFrame,
    *,
    input_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pitcher_source_path = input_dir / "pitcher_p_damage.parquet"
    pitch_type_source_path = input_dir / "pitch_types_p_damage.parquet"

    if pitcher_source_path.exists():
        pitcher_source = pl.read_parquet(pitcher_source_path).rename({"pitcher_name": "name"})
        pitcher_keys = ["pitcher_mlbid", "season"]
        if "level_id" in pitcher_source.columns:
            pitcher_keys.append("level_id")
        pitcher_source = _weighted_pct_merge(
            pitcher_source,
            keys=pitcher_keys,
            weight_col="n",
            value_cols=["p_Damage_pct", "p_Damage_with_loc_pct"],
        ).rename({"n": "p_Damage_n"})
        pitchers = pitchers.join(
            pitcher_source,
            on=pitcher_keys,
            how="left",
        )

    if pitch_type_source_path.exists():
        pitch_type_source = pl.read_parquet(pitch_type_source_path).rename({"pitcher_name": "name"})
        pitch_type_keys = ["pitcher_mlbid", "season", "pitch_tag"]
        if "level_id" in pitch_type_source.columns:
            pitch_type_keys.append("level_id")
        pitch_type_source = _weighted_pct_merge(
            pitch_type_source,
            keys=pitch_type_keys,
            weight_col="n",
            value_cols=["p_Damage_pct", "p_Damage_with_loc_pct"],
        ).rename({"n": "p_Damage_n"})
        pitch_types = pitch_types.join(
            pitch_type_source,
            on=pitch_type_keys,
            how="left",
        )

    if "p_Damage_pct" not in pitchers.columns:
        pitchers = pitchers.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("p_Damage_pct"),
                pl.lit(None, dtype=pl.Float64).alias("p_Damage_with_loc_pct"),
            ]
        )
    if "p_Damage_pct" not in pitch_types.columns:
        pitch_types = pitch_types.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("p_Damage_pct"),
                pl.lit(None, dtype=pl.Float64).alias("p_Damage_with_loc_pct"),
            ]
        )

    return pitchers, pitch_types


def _fetch_mlb_sb_stats(
    seasons: list[int],
    sport_id: int,
    group: str = "hitting",
) -> pl.DataFrame:
    """
    Fetch season SB and CS per player from the MLB Stats API for a given sportId.

    Uses a single paginated request per season via playerPool=All. Aggregates
    across teams for mid-season trades. Returns an empty DataFrame on any
    network or parse failure so the caller can degrade gracefully.

    Args:
        seasons:  List of seasons to fetch.
        sport_id: MLB Stats API sportId (1=MLB, 11=AAA, etc.).
        group:    API stat group — "hitting" (runner SB/CS) or "pitching"
                  (stolen bases against). Controls both the API parameter and
                  the output player-id column name (runner_mlbid vs pitcher_mlbid).
    """
    player_col = "runner_mlbid" if group == "hitting" else "pitcher_mlbid"
    caller = "build_baserunning" if group == "hitting" else "build_pitcher_baserunning"

    try:
        import requests as _requests
        import time as _time
    except ImportError:
        print(f"{caller}: requests not available, SB/CS will be null.")
        return pl.DataFrame(schema={
            player_col: pl.Float64, "season": pl.Int32,
            "SB": pl.Int32, "CS": pl.Int32,
        })

    records: dict[tuple, dict] = {}
    for season in seasons:
        offset = 0
        limit = 500
        while True:
            try:
                resp = _requests.get(
                    "https://statsapi.mlb.com/api/v1/stats",
                    params={
                        "stats": "season", "group": group,
                        "season": season, "sportId": sport_id, "gameType": "R",
                        "playerPool": "All", "limit": limit, "offset": offset,
                    },
                    timeout=15,
                )
                splits = resp.json().get("stats", [{}])[0].get("splits", [])
            except Exception as exc:
                print(f"{caller}: API error (sportId={sport_id}, season={season}): {exc}")
                break
            if not splits:
                break
            for s in splits:
                pid = float(s["player"]["id"])
                stat = s.get("stat", {})
                key = (pid, season)
                if key in records:
                    records[key]["SB"] += int(stat.get("stolenBases", 0))
                    records[key]["CS"] += int(stat.get("caughtStealing", 0))
                else:
                    records[key] = {
                        player_col: pid,
                        "season": season,
                        "SB": int(stat.get("stolenBases", 0)),
                        "CS": int(stat.get("caughtStealing", 0)),
                    }
            if len(splits) < limit:
                break
            offset += limit
            _time.sleep(0.1)

    if not records:
        return pl.DataFrame(schema={
            player_col: pl.Float64, "season": pl.Int32,
            "SB": pl.Int32, "CS": pl.Int32,
        })
    return pl.DataFrame(list(records.values()))


def build_baserunning(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-runner baserunning stats: SBO, SB, CS, takeoff_rate.

    SBO: plate appearances where runner occupied a stealable base (1st with 2nd
         empty, 2nd with 3rd empty, or 3rd base). Derived from mlbapi.baseout
         base state columns in the pitch data.

    SB / CS: fetched from the MLB Stats API (MLB regular season only). Will be
         null for minor league rows or non-regular-season game type groups.

    takeoff_rate: (SB + CS) / SBO — measures baserunner aggression.
    """
    required = {
        "firstbase_pre", "secondbase_pre", "thirdbase_pre",
        "outs_pre", "batter_mlbid", "game_pk", "at_bat_index", "event_index",
        "season", "level_id", "game_type_group",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"build_baserunning: missing columns {sorted(missing)}, returning empty.")
        return pl.DataFrame()

    group_cols = ["season", "level_id", "game_type_group"]

    # One row per PA: baseout data lands on the pitch with the highest event_index.
    # Use .over() to select that row reliably — group_by().last() does not respect
    # a prior sort() in Polars.
    pa = (
        df.filter(
            pl.col("event_index") == pl.col("event_index").max().over(["game_pk", "at_bat_index"])
        )
        .unique(subset=["game_pk", "at_bat_index"])
        .filter(pl.col("outs_pre").is_not_null())
    )

    # ── SBO ──────────────────────────────────────────────────────────────────
    sbo_parts = [
        pa.filter(
            pl.col("firstbase_pre").is_not_null() & pl.col("secondbase_pre").is_null()
        ).rename({"firstbase_pre": "runner_mlbid"}),
        pa.filter(
            pl.col("secondbase_pre").is_not_null() & pl.col("thirdbase_pre").is_null()
        ).rename({"secondbase_pre": "runner_mlbid"}),
        pa.filter(
            pl.col("thirdbase_pre").is_not_null()
        ).rename({"thirdbase_pre": "runner_mlbid"}),
    ]
    sbo = (
        pl.concat([p.select(["runner_mlbid"] + group_cols) for p in sbo_parts])
        .group_by(["runner_mlbid"] + group_cols)
        .agg(pl.len().alias("SBO"))
    )

    # ── SB / CS from MLB Stats API ────────────────────────────────────────────
    # Fetch per (season, level_id), treating level_id as the API's sportId.
    # Join on all four group keys so each level's SB/CS stays isolated.
    # Non-regular-season rows receive nulls via the game_type_group constant.
    seasons = sorted(df["season"].unique().to_list())
    level_ids = sorted(df["level_id"].unique().to_list())
    print(
        f"build_baserunning: fetching SB/CS from MLB Stats API "
        f"for seasons {seasons}, levels {level_ids}..."
    )
    api_parts = []
    for level_id in level_ids:
        part = _fetch_mlb_sb_stats(seasons, sport_id=level_id)
        if not part.is_empty():
            part = part.with_columns([
                pl.lit(level_id).cast(pl.Int32).alias("level_id"),
                pl.lit("Regular Season").alias("game_type_group"),
            ])
            api_parts.append(part)

    api_stats = pl.concat(api_parts) if api_parts else pl.DataFrame()

    if not api_stats.is_empty():
        # Deduplicate in case a player appears under multiple teams at the same level
        api_stats = (
            api_stats.group_by(["runner_mlbid", "season", "level_id", "game_type_group"])
            .agg([pl.col("SB").sum(), pl.col("CS").sum()])
        )

    # ── Combine and compute takeoff_rate ─────────────────────────────────────
    join_keys = ["runner_mlbid", "season", "level_id", "game_type_group"]
    result = sbo.join(api_stats, on=join_keys, how="left") if not api_stats.is_empty() else (
        sbo.with_columns([
            pl.lit(None).cast(pl.Int32).alias("SB"),
            pl.lit(None).cast(pl.Int32).alias("CS"),
        ])
    )
    result = result.with_columns([
        (pl.col("SB") + pl.col("CS")).alias("takeoff_rate_num"),
        pl.col("SBO").alias("takeoff_rate_den"),
        ((pl.col("SB") + pl.col("CS")) / pl.col("SBO")).alias("takeoff_rate"),
    ])

    # ── Player names from batter lookup ──────────────────────────────────────
    names = (
        df.select(["batter_mlbid", "batter_name_first", "batter_name_last"])
        .unique(subset=["batter_mlbid"])
        .with_columns([
            (pl.col("batter_name_first") + " " + pl.col("batter_name_last")).alias("runner_name"),
            pl.col("batter_mlbid").cast(pl.Float64).alias("runner_mlbid"),
        ])
        .select(["runner_mlbid", "runner_name"])
    )

    return result.join(names, on="runner_mlbid", how="left")


def build_pitcher_baserunning(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-pitcher baserunning-against stats: SBO, SB, CS, takeoff_rate.

    SBO: plate appearances where a runner occupied a stealable base while the
         pitcher was on the mound. Derived from the same base-state columns as
         build_baserunning() but grouped by pitcher_mlbid.

    SB / CS: fetched from the MLB Stats API with group=pitching (stolen bases
         allowed). Will be null for minor league rows or non-regular-season
         game type groups.

    takeoff_rate: (SB + CS) / SBO — measures baserunner aggression against
         this pitcher. Higher = worse (opponents run more).
    """
    required = {
        "firstbase_pre", "secondbase_pre", "thirdbase_pre",
        "outs_pre", "pitcher_mlbid", "game_pk", "at_bat_index", "event_index",
        "season", "level_id", "game_type_group",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"build_pitcher_baserunning: missing columns {sorted(missing)}, returning empty.")
        return pl.DataFrame()

    group_cols = ["season", "level_id", "game_type_group"]

    # One row per PA: take pitch with the highest event_index per (game_pk, at_bat_index)
    pa = (
        df.filter(
            pl.col("event_index") == pl.col("event_index").max().over(["game_pk", "at_bat_index"])
        )
        .unique(subset=["game_pk", "at_bat_index"])
        .filter(pl.col("outs_pre").is_not_null())
    )

    # ── SBO ──────────────────────────────────────────────────────────────────
    # Cast pitcher_mlbid to Float64 to align with API-fetched player IDs (f64).
    pa = pa.with_columns(pl.col("pitcher_mlbid").cast(pl.Float64))
    sbo_parts = [
        pa.filter(
            pl.col("firstbase_pre").is_not_null() & pl.col("secondbase_pre").is_null()
        ),
        pa.filter(
            pl.col("secondbase_pre").is_not_null() & pl.col("thirdbase_pre").is_null()
        ),
        pa.filter(
            pl.col("thirdbase_pre").is_not_null()
        ),
    ]
    sbo = (
        pl.concat([p.select(["pitcher_mlbid"] + group_cols) for p in sbo_parts])
        .group_by(["pitcher_mlbid"] + group_cols)
        .agg(pl.len().alias("SBO"))
    )

    # ── SB / CS from MLB Stats API (pitching side) ────────────────────────────
    seasons = sorted(df["season"].unique().to_list())
    level_ids = sorted(df["level_id"].unique().to_list())
    print(
        f"build_pitcher_baserunning: fetching SB/CS-against from MLB Stats API "
        f"for seasons {seasons}, levels {level_ids}..."
    )
    api_parts = []
    for level_id in level_ids:
        part = _fetch_mlb_sb_stats(seasons, sport_id=level_id, group="pitching")
        if not part.is_empty():
            part = part.with_columns([
                pl.lit(level_id).cast(pl.Int32).alias("level_id"),
                pl.lit("Regular Season").alias("game_type_group"),
            ])
            api_parts.append(part)

    api_stats = pl.concat(api_parts) if api_parts else pl.DataFrame()

    if not api_stats.is_empty():
        api_stats = (
            api_stats.group_by(["pitcher_mlbid", "season", "level_id", "game_type_group"])
            .agg([pl.col("SB").sum(), pl.col("CS").sum()])
        )

    # ── Combine and compute takeoff_rate ─────────────────────────────────────
    join_keys = ["pitcher_mlbid", "season", "level_id", "game_type_group"]
    result = sbo.join(api_stats, on=join_keys, how="left") if not api_stats.is_empty() else (
        sbo.with_columns([
            pl.lit(None).cast(pl.Int32).alias("SB"),
            pl.lit(None).cast(pl.Int32).alias("CS"),
        ])
    )
    result = result.with_columns([
        (pl.col("SB") + pl.col("CS")).alias("takeoff_rate_num"),
        pl.col("SBO").alias("takeoff_rate_den"),
        ((pl.col("SB") + pl.col("CS")) / pl.col("SBO")).alias("takeoff_rate"),
    ])

    # ── Pitcher names ─────────────────────────────────────────────────────────
    names = (
        df.select(["pitcher_mlbid", "name"])
        .unique(subset=["pitcher_mlbid"])
        .with_columns(
            pl.col("pitcher_mlbid").cast(pl.Float64),
        )
    )

    return result.join(names, on="pitcher_mlbid", how="left")


def _build_outputs(
    pitch: pl.DataFrame,
    min_season: int,
    max_season: int,
    *,
    input_dir: Path,
) -> dict[str, pl.DataFrame]:
    # Normalize names so accents don't split groupings across seasons
    pitch = _normalize_player_names(pitch)
    pitch = _normalize_age_columns(pitch)
    # Tag pitches for aggregation
    pitch = _tag_pitch(pitch)
    # Derive game_type_group for all downstream groupings
    pitch = _add_game_type_group(pitch)

    # Warn if stuff_raw is missing for a material share of pitches — indicates
    # the stuff model was not run on some rows (e.g. incremental data pulled
    # without applying models). Stuff grades will be null for those pitchers.
    if "stuff_raw" in pitch.columns:
        total = len(pitch)
        null_count = pitch["stuff_raw"].is_null().sum()
        if total > 0:
            null_pct = null_count / total * 100
            if null_pct > 1.0:
                print(
                    f"WARNING: {null_count:,} of {total:,} pitches ({null_pct:.1f}%) "
                    f"have null stuff_raw — stuff model may not have been run on all rows. "
                    f"Affected pitchers will have null Pitch Grades."
                )
    else:
        print("WARNING: stuff_raw column missing — Pitch Grades will not be computed.")

    # Compute stuff grade percentiles from pitcher-level averages
    print(f"Computing stuff percentiles...{_mem_note()}")
    stuff_percentiles = compute_stuff_percentiles(pitch, min_pitches=50)

    # Build aggregated tables
    print(f"Building hitters...{_mem_note()}")
    hitters = build_hitters(pitch)

    print(f"Building pitchers...{_mem_note()}")
    pitchers = build_pitchers(pitch)

    print(f"Building pitch types...{_mem_note()}")
    pitch_types = build_pitch_types(pitch)

    print(f"Building team hitting...{_mem_note()}")
    team_hitting = build_team_hitting(pitch)

    print(f"Building team pitching...{_mem_note()}")
    team_pitching = build_team_pitching(pitch)

    print(f"Building league hitting...{_mem_note()}")
    league_hitting = build_league_hitting(pitch)

    print(f"Building league pitching...{_mem_note()}")
    league_pitching = build_league_pitching(pitch)

    print(f"Building park data...{_mem_note()}")
    park_data = build_park_data(pitch)

    print(f"Building baserunning...{_mem_note()}")
    baserunning = build_baserunning(pitch)

    print(f"Building pitcher baserunning...{_mem_note()}")
    pitcher_baserunning = build_pitcher_baserunning(pitch)

    print("Building league pitch types...")
    league_pitch_types_shapes = build_league_pitch_types(pitch)
    expected_cols = {
        "season",
        "throws",
        "pitch_tag",
        "pct",
        "velo",
        "vaa",
        "haa",
        "vbreak",
        "hbreak",
        "SwStr",
        "LA_lte_0",
        "Z_Contact",
        "Zone",
        "Ball_pct",
        "Chase",
        "CSW",
    }
    missing_cols = expected_cols - set(league_pitch_types_shapes.columns)
    if missing_cols:
        raise ValueError(
            f"league_pitch_types_shapes missing columns: {sorted(missing_cols)}"
        )

    print("Building splits...")
    hitter_splits = build_hitter_splits(pitch)
    pitcher_splits, pitch_type_splits = build_pitching_splits(pitch, stuff_percentiles)

    # Apply stuff grades to pitch_types
    pitch_types = apply_stuff_grade(pitch_types, stuff_percentiles)

    # For pitchers, compute weighted average stuff grade from pitch types
    pitcher_stuff = (
        pitch_types.filter(pl.col("stuff").is_not_null())
        .group_by(["pitcher_mlbid", "season", "level_id", "game_type_group"])
        .agg((pl.col("stuff") * pl.col("pitches")).sum() / pl.col("pitches").sum())
        .rename({"stuff": "stuff_grade"})
    )
    pitchers = (
        pitchers.join(
            pitcher_stuff, on=["pitcher_mlbid", "season", "level_id", "game_type_group"], how="left"
        )
        .with_columns(pl.col("stuff_grade").alias("stuff"))
        .drop("stuff_grade")
    )

    pitchers, pitch_types = _apply_p_swstr_sources(
        pitchers,
        pitch_types,
        input_dir=input_dir,
    )
    pitchers, pitch_types = _apply_p_damage_sources(
        pitchers,
        pitch_types,
        input_dir=input_dir,
    )

    # For team_pitching, compute stuff grades from raw pitch data
    team_pitch_types = pitch.group_by(
        ["pitching_code", "season", "level_id", "game_type_group", "pitch_tag"]
    ).agg(
        [
            pl.mean("stuff_raw").alias("stuff_raw"),
            pl.len().alias("pitches"),
        ]
    )
    team_pitch_types = apply_stuff_grade(team_pitch_types, stuff_percentiles)
    team_stuff = (
        team_pitch_types.filter(pl.col("stuff").is_not_null())
        .group_by(["pitching_code", "season", "level_id", "game_type_group"])
        .agg((pl.col("stuff") * pl.col("pitches")).sum() / pl.col("pitches").sum())
        .rename({"stuff": "stuff_grade"})
    )
    team_pitching = (
        team_pitching.join(
            team_stuff, on=["pitching_code", "season", "level_id", "game_type_group"], how="left"
        )
        .with_columns(pl.col("stuff_grade").alias("stuff"))
        .drop("stuff_grade")
    )

    league_pitch_types_stuff = pitch.group_by(["season", "level_id", "game_type_group", "pitch_tag"]).agg(
        [
            pl.mean("stuff_raw").alias("stuff_raw"),
            pl.len().alias("pitches"),
        ]
    )
    league_pitch_types_stuff = apply_stuff_grade(
        league_pitch_types_stuff, stuff_percentiles
    )
    league_stuff = (
        league_pitch_types_stuff.filter(pl.col("stuff").is_not_null())
        .group_by(["season", "level_id", "game_type_group"])
        .agg((pl.col("stuff") * pl.col("pitches")).sum() / pl.col("pitches").sum())
        .rename({"stuff": "stuff_grade"})
    )
    league_pitching = (
        league_pitching.join(league_stuff, on=["season", "level_id", "game_type_group"], how="left")
        .with_columns(pl.col("stuff_grade").alias("stuff"))
        .drop("stuff_grade")
    )

    # Add percentiles
    print(f"Computing hitter percentiles...{_mem_note()}")
    hitter_pct = add_percentiles(
        hitters,
        group_cols=["season", "level_id", "game_type_group"],
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
            "whiffs_vs_95",
            "contact_vs_avg",
        ],
        filter_col="PA",
        min_threshold=200,
    )

    print(f"Computing pitcher percentiles...{_mem_note()}")
    _pitcher_value_cols = [
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
        "p_SwStr_pct",
        "Damage_pct",
        "p_Damage_pct",
    ]
    _pitcher_value_cols = [col for col in _pitcher_value_cols if col in pitchers.columns]
    pitcher_pct = add_percentiles(
        pitchers,
        group_cols=["season", "level_id", "game_type_group"],
        value_cols=_pitcher_value_cols,
        filter_col="IP",
        min_threshold=40,
    )

    pitch_types_value_cols = [
        "pct",
        "stuff",
        "velo",
        "max_velo",
        "vaa",
        "haa",
        "vbreak",
        "hbreak",
        "SwStr",
        "LA_lte_0",
        "Ball_pct",
        "Z_Contact",
        "Chase",
        "CSW",
        "p_SwStr_pct",
        "Damage_pct",
        "p_Damage_pct",
    ]
    pitch_types_value_cols = [
        col for col in pitch_types_value_cols if col in pitch_types.columns
    ]

    print(f"Computing pitch type percentiles...{_mem_note()}")
    pitch_types_pct = add_percentiles(
        pitch_types,
        group_cols=["season", "level_id", "game_type_group", "pitch_tag"],
        value_cols=pitch_types_value_cols,
        filter_col="pitches",
        min_threshold=100,
    )

    return {
        f"damage_pos_{min_season}_{max_season}.parquet": hitters,
        "pitcher_stuff_new.parquet": pitchers,
        "new_pitch_types.parquet": pitch_types,
        "new_team_damage.parquet": team_hitting,
        "new_team_stuff.parquet": team_pitching,
        "league_pitch_types.parquet": league_pitch_types_shapes,
        "hitter_splits.parquet": hitter_splits,
        "pitcher_splits.parquet": pitcher_splits,
        "pitch_types_splits.parquet": pitch_type_splits,
        "hitter_pctiles.parquet": hitter_pct,
        "pitcher_pctiles.parquet": pitcher_pct,
        "pitch_types_pctiles.parquet": pitch_types_pct,
        "new_hitting_lg_avg.parquet": league_hitting,
        "new_lg_stuff.parquet": league_pitching,
        "park_data.parquet": park_data,
        "baserunning.parquet": baserunning,
        "pitcher_baserunning.parquet": pitcher_baserunning,
    }


def _mem_note() -> str:
    if psutil is None:
        return ""
    try:
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / (1024**3)
        return f" | RSS {rss_gb:0.2f} GB"
    except Exception:
        return ""


def main(
    parquet_path: Path,
    out_dir: Path,
    min_season: int,
    max_season: int,
    chunk_by_season: bool = False,
    chunk_dir: Path | None = None,
    parquet_path_level1: Path | None = None,
    parquet_path_no_level1: Path | None = None,
    input_dir: Path | None = None,
) -> None:
    """Read parquet data and generate aggregated parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = input_dir or out_dir
    if not chunk_by_season:
        if parquet_path is not None:
            print(f"Reading pitch data from {parquet_path}...")
            pitch = pl.read_parquet(parquet_path)
        else:
            if parquet_path_level1 is None:
                parquet_path_level1 = (
                    input_dir / f"pitch_data_{min_season}_{max_season}_level1.parquet"
                )
            if parquet_path_no_level1 is None:
                parquet_path_no_level1 = (
                    input_dir / f"pitch_data_2021_{max_season}_no_level1.parquet"
                )
            if not parquet_path_level1.exists():
                raise FileNotFoundError(
                    f"Input parquet not found: {parquet_path_level1}"
                )
            if not parquet_path_no_level1.exists():
                raise FileNotFoundError(
                    f"Input parquet not found: {parquet_path_no_level1}"
                )
            print(f"Reading pitch data from {parquet_path_level1}...")
            pitch_level1 = pl.read_parquet(parquet_path_level1)
            print(f"Reading pitch data from {parquet_path_no_level1}...")
            pitch_no_level1 = pl.read_parquet(parquet_path_no_level1)
            pitch = pl.concat([pitch_level1, pitch_no_level1], how="diagonal_relaxed")
        print(f"Loaded {len(pitch):,} pitch rows.")

        outputs = _build_outputs(
            pitch,
            min_season,
            max_season,
            input_dir=input_dir,
        )
        print(f"Writing parquet files to {out_dir}...{_mem_note()}")
        for name, df in outputs.items():
            write_parquet(df, name, out_dir)
        print("Aggregation complete!")
        return

    if chunk_dir is None:
        chunk_dir = out_dir / "_season_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"Chunking by season into {chunk_dir}...")
    if parquet_path is not None:
        scan = pl.scan_parquet(parquet_path)
    else:
        if parquet_path_level1 is None:
            parquet_path_level1 = (
                input_dir / f"pitch_data_{min_season}_{max_season}_level1.parquet"
            )
        if parquet_path_no_level1 is None:
            parquet_path_no_level1 = (
                input_dir / f"pitch_data_2021_{max_season}_no_level1.parquet"
            )
        if not parquet_path_level1.exists():
            raise FileNotFoundError(f"Input parquet not found: {parquet_path_level1}")
        if not parquet_path_no_level1.exists():
            raise FileNotFoundError(
                f"Input parquet not found: {parquet_path_no_level1}"
            )
        scan_level1 = pl.scan_parquet(parquet_path_level1)
        scan_no_level1 = pl.scan_parquet(parquet_path_no_level1)
        scan = pl.concat([scan_level1, scan_no_level1], how="diagonal_relaxed")

    chunk_map: dict[str, list[Path]] = {}
    final_damage_name = f"damage_pos_{min_season}_{max_season}.parquet"
    for season in range(min_season, max_season + 1):
        print(f"Processing season {season}...{_mem_note()}")
        pitch = scan.filter(pl.col("season") == season).collect()
        print(f"Loaded {len(pitch):,} pitch rows for {season}.")
        outputs = _build_outputs(
            pitch,
            season,
            season,
            input_dir=input_dir,
        )
        for name, df in outputs.items():
            final_name = (
                final_damage_name
                if name.startswith("damage_pos_")
                else name
            )
            stem = Path(final_name).stem
            chunk_name = f"{stem}.season_{season}.parquet"
            chunk_path = chunk_dir / chunk_name
            write_parquet(df, chunk_name, chunk_dir)
            chunk_map.setdefault(final_name, []).append(chunk_path)

    print(f"Combining season chunks into {out_dir}...{_mem_note()}")
    for final_name, parts in chunk_map.items():
        lazy_frames = [pl.scan_parquet(p) for p in parts]
        combined = pl.concat(lazy_frames, how="diagonal_relaxed")
        combined.sink_parquet(out_dir / final_name)
        print(f"Wrote {final_name} from {len(parts)} chunks")

    print("Aggregation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate pitch data into parquet files")
    parser.add_argument("--min-season", type=int, default=2015)
    parser.add_argument("--max-season", type=int, default=2025)
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Path to input parquet file (default: pitch_data_{min}_{max}.parquet)",
    )
    parser.add_argument(
        "--parquet-path-level1",
        type=Path,
        default=None,
        help="Path to level 1 parquet file (default: pitch_data_{min}_{max}_level1.parquet)",
    )
    parser.add_argument(
        "--parquet-path-no-level1",
        type=Path,
        default=None,
        help="Path to non-level1 parquet file (default: pitch_data_2021_{max}_no_level1.parquet)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        help="Input directory for default parquet files (defaults to data/raw)",
    )
    parser.add_argument(
        "--chunk-by-season",
        action="store_true",
        help="Process one season at a time to reduce memory usage",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Directory to store per-season chunk outputs (default: <out-dir>/_season_chunks)",
    )
    args = parser.parse_args()

    if args.parquet_path is not None and not args.parquet_path.exists():
        raise FileNotFoundError(
            "Input parquet not found. Provide --parquet-path or ensure the file exists: "
            f"{args.parquet_path}"
        )

    main(
        parquet_path=args.parquet_path,
        out_dir=args.out_dir,
        min_season=args.min_season,
        max_season=args.max_season,
        chunk_by_season=args.chunk_by_season,
        chunk_dir=args.chunk_dir,
        parquet_path_level1=args.parquet_path_level1,
        parquet_path_no_level1=args.parquet_path_no_level1,
        input_dir=args.input_dir,
    )
