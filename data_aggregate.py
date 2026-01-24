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
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR

STUFF_SCALE_MEAN = 50.0
STUFF_SCALE_STD = 10.0


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


def compute_stuff_percentiles(
    df: pl.DataFrame,
    raw_col: str = "stuff_raw",
    min_pitches: int = 50,
) -> pl.DataFrame:
    """Compute stuff grade percentile thresholds by season + pitch_tag."""
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
            .cast(pl.Int64)
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
) -> pl.DataFrame:
    """Add percentile columns for value_cols, grouped by group_cols."""
    if df.is_empty():
        return df
    df_pd = df.to_pandas()
    group_list = list(group_cols)
    value_list = list(value_cols)

    if filter_col and min_threshold is not None and filter_col in df_pd.columns:
        qualified = df_pd[df_pd[filter_col] >= min_threshold].copy()
        if qualified.empty:
            for col in value_list:
                df_pd[f"{col}_pctile"] = None
            return pl.from_pandas(df_pd)

        pct = qualified.groupby(group_list)[value_list].rank(pct=True) * 100
        pct.columns = [f"{c}_pctile" for c in pct.columns]
        pct.index = qualified.index

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
    print(f"Wrote {len(df):,} rows to {path}")


def main(
    parquet_path: Path,
    out_dir: Path,
    min_season: int,
    max_season: int,
) -> None:
    """Read parquet data and generate aggregated CSV files."""
    print(f"Reading pitch data from {parquet_path}...")
    pitch = pl.read_parquet(parquet_path)
    print(f"Loaded {len(pitch):,} pitch rows.")

    # Tag pitches for aggregation
    pitch = _tag_pitch(pitch)

    # Compute stuff grade percentiles from pitcher-level averages
    stuff_percentiles = compute_stuff_percentiles(pitch, min_pitches=50)

    # Build aggregated tables
    print("Building hitters...")
    hitters = build_hitters(pitch)

    print("Building pitchers...")
    pitchers = build_pitchers(pitch)

    print("Building pitch types...")
    pitch_types = build_pitch_types(pitch)

    print("Building team hitting...")
    team_hitting = build_team_hitting(pitch)

    print("Building team pitching...")
    team_pitching = build_team_pitching(pitch)

    # Apply stuff grades to pitch_types
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

    # For team_pitching, compute stuff grades from raw pitch data
    team_pitch_types = (
        pitch.group_by(["pitching_code", "season", "level_id", "pitch_tag"])
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

    # Write CSV files
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(hitters, f"damage_pos_{min_season}_{max_season}.csv", out_dir)
    write_csv(pitchers, "pitcher_stuff_new.csv", out_dir)
    write_csv(pitch_types, "new_pitch_types.csv", out_dir)
    write_csv(team_hitting, "new_team_damage.csv", out_dir)
    write_csv(team_pitching, "new_team_stuff.csv", out_dir)

    # Add percentiles
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

    print("Aggregation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate pitch data into CSVs")
    parser.add_argument("--min-season", type=int, default=2015)
    parser.add_argument("--max-season", type=int, default=2025)
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Path to input parquet file (default: pitch_data_{min}_{max}.parquet)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.getenv("POOBAH_OUT_DIR", OUT_DIR)),
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    if args.parquet_path is None:
        args.parquet_path = (
            args.out_dir / f"pitch_data_{args.min_season}_{args.max_season}.parquet"
        )

    main(
        parquet_path=args.parquet_path,
        out_dir=args.out_dir,
        min_season=args.min_season,
        max_season=args.max_season,
    )
