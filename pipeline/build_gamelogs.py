# -*- coding: utf-8 -*-
"""
Gamelog Aggregation Script

Reads pitch-level parquet data and produces game-by-game aggregates for
hitters, pitchers, and individual pitch types.

Usage:
    python build_gamelogs.py --parquet-path data/raw/pitch_data_2026.parquet
    python build_gamelogs.py --parquet-path data/raw/pitch_data_2026.parquet \\
        --chunk-by-season --chunk-dir data/output/_season_chunks
    python build_gamelogs.py --parquet-path data/raw/pitch_data_2026.parquet \\
        --scored-path data/raw/location_v13_scored.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Optional

import polars as pl

_REPO_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = _REPO_DIR / "data" / "output" / "_season_chunks"
OUT_DIR   = _REPO_DIR / "data" / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_game_type_group(df: pl.DataFrame) -> pl.DataFrame:
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


def _damage_expr() -> pl.Expr:
    return (
        (pl.col("is_in_play") == True)
        & pl.col("damage_pred").is_not_null()
        & (pl.col("exit_velo") >= pl.col("damage_pred"))
        & (pl.col("launch_angle") > 0)
        & (pl.col("spray_angle_adj") >= -50)
        & (pl.col("spray_angle_adj") <= 50)
    )


def _bbe_expr() -> pl.Expr:
    return pl.col("bbe") == 1


def _common_aggs(include_tbf: bool = False, include_pa: bool = False) -> list[pl.Expr]:
    bbe  = _bbe_expr()
    dmg  = _damage_expr()
    exprs = []
    if include_pa:
        exprs.append(pl.n_unique("pa_id").alias("PA"))
    if include_tbf:
        exprs.append(pl.n_unique("pa_id").alias("TBF"))
        exprs.append(pl.len().alias("pitches"))
    exprs += [
        pl.sum("bbe").alias("bbe"),
        dmg.sum().alias("damaged_bbe"),
        (pl.col("pitch_outcome") == "HR").sum().alias("HR"),
        pl.col("pitch_outcome").is_in(["2B", "3B", "HR"]).sum().alias("XBH"),
        pl.col("pitch_outcome").is_in(["1B", "2B", "3B", "HR"]).sum().alias("hits"),
        (bbe & (pl.col("launch_angle") <= 0)).sum().alias("la_lte_0_bbe"),
        (bbe & (pl.col("launch_angle") >= 20)).sum().alias("la_gte_20_bbe"),
        (pl.col("swing") == 1).sum().alias("swings"),
        ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum().alias("chases"),
        (pl.col("whiff") == 1).sum().alias("whiffs"),
        pl.sum("bb").alias("BB"),
        pl.sum("k").alias("K"),
    ]
    return exprs


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_hitter_gamelogs(df: pl.DataFrame) -> pl.DataFrame:
    keys = [c for c in [
        "batter_mlbid", "hitter_name", "hitting_code",
        "game_pk", "game_date", "season", "level_id", "game_type", "game_type_group",
    ] if c in df.columns]
    hitter_aggs = _common_aggs(include_pa=True)
    hitter_aggs += [
        pl.len().alias("pitches"),
        pl.col("pitching_code").first().alias("opp_team"),
        (
            (pl.col("launch_angle") >= 20)
            & (pl.col("spray_angle_adj") < -15)
            & (pl.col("is_in_play") == True)
        ).sum().alias("pulled_fbs"),
        ((pl.col("decision_value") > 0) & (pl.col("swing") == 0)).sum().alias("selective_takes"),
        ((pl.col("decision_value") < 0) & (pl.col("swing") == 0)).sum().alias("hittable_takes"),
        (pl.col("pitch_group") == "FA").sum().alias("FA"),
        (pl.col("pitch_group") == "BR").sum().alias("BR"),
        (pl.col("pitch_group") == "OFF").sum().alias("OFF"),
        (pl.col("pitcher_hand") == "R").sum().alias("vs_RHP"),
        (pl.col("pitcher_hand") == "L").sum().alias("vs_LHP"),
    ]
    return (
        df.group_by(keys)
        .agg(hitter_aggs)
        .sort(["season", "game_date", "batter_mlbid"])
    )


def build_pitcher_gamelogs(df: pl.DataFrame) -> pl.DataFrame:
    keys = [c for c in [
        "pitcher_mlbid", "pitcher_name", "pitching_code", "pitcher_hand",
        "game_pk", "game_date", "season", "level_id", "game_type", "game_type_group",
    ] if c in df.columns]
    pitcher_aggs = _common_aggs(include_tbf=True)
    pitcher_aggs += [
        (pl.col("is_inzone_pi") == True).sum().alias("zone_pitches"),
        (pl.col("pitch_group") == "FA").sum().alias("FA"),
        (pl.col("pitch_group") == "BR").sum().alias("BR"),
        (pl.col("pitch_group") == "OFF").sum().alias("OFF"),
        pl.col("hitting_code").first().alias("opp_team"),
        pl.col("is_strike").sum().alias("strikes"),
        pl.col("is_ball").sum().alias("balls"),
        (pl.col("is_inzone_pi") == False).sum().alias("out_of_zone"),
        (pl.col("batter_hand") == "L").sum().alias("vs_LHB"),
        (pl.col("batter_hand") == "R").sum().alias("vs_RHB"),
        pl.col("pitch_velo")
            .filter(pl.col("pitch_group") == "FA")
            .mean()
            .round(1)
            .alias("FA_mph"),
    ]
    if "stuff_raw" in df.columns:
        pitcher_aggs.append(
            pl.col("stuff_raw").mean().alias("stuff_raw")
        )
    return (
        df.group_by(keys)
        .agg(pitcher_aggs)
        .sort(["season", "game_date", "pitcher_mlbid"])
    )


def build_pitch_type_gamelogs(df: pl.DataFrame) -> pl.DataFrame:
    keys = [c for c in [
        "pitcher_mlbid", "pitcher_name", "pitching_code", "pitcher_hand",
        "game_pk", "game_date", "season", "level_id", "game_type", "game_type_group",
        "pitch_tag",
    ] if c in df.columns]
    pt_aggs = [pl.len().alias("pitches")]
    pt_aggs += [
        pl.sum("bbe").alias("bbe"),
        _damage_expr().sum().alias("damaged_bbe"),
        (pl.col("pitch_outcome") == "HR").sum().alias("HR"),
        pl.col("pitch_outcome").is_in(["2B", "3B", "HR"]).sum().alias("XBH"),
        pl.col("pitch_outcome").is_in(["1B", "2B", "3B", "HR"]).sum().alias("hits"),
        (_bbe_expr() & (pl.col("launch_angle") <= 0)).sum().alias("la_lte_0_bbe"),
        (_bbe_expr() & (pl.col("launch_angle") >= 20)).sum().alias("la_gte_20_bbe"),
        (pl.col("swing") == 1).sum().alias("swings"),
        ((pl.col("swing") == 1) & (pl.col("is_inzone_pi") == False)).sum().alias("chases"),
        (pl.col("whiff") == 1).sum().alias("whiffs"),
        pl.sum("bb").alias("BB"),
        pl.sum("k").alias("K"),
        (pl.col("is_inzone_pi") == True).sum().alias("zone_pitches"),
        pl.col("hitting_code").first().alias("opp_team"),
        pl.col("is_strike").sum().alias("strikes"),
        pl.col("is_ball").sum().alias("balls"),
        (pl.col("is_inzone_pi") == False).sum().alias("out_of_zone"),
        (pl.col("batter_hand") == "L").sum().alias("vs_LHB"),
        (pl.col("batter_hand") == "R").sum().alias("vs_RHB"),
        pl.col("pitch_velo").mean().round(1).alias("velo"),
    ]
    if "stuff_raw" in df.columns:
        pt_aggs.append(pl.col("stuff_raw").mean().alias("stuff_raw"))
    return (
        df.group_by(keys)
        .agg(pt_aggs)
        .sort(["season", "game_date", "pitcher_mlbid", "pitch_tag"])
    )


def _merge_scored_pred_v13(
    pitchers: pl.DataFrame,
    pit_types: pl.DataFrame,
    scored_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load location_v13_scored.parquet (must have game_pk + pred_v13) and
    merge game-level mean pred_v13 onto pitcher and pitch-type gamelogs.
    """
    schema = pl.read_parquet_schema(scored_path)
    if "game_pk" not in schema or "pred_v13" not in schema:
        print(f"  scored parquet missing game_pk or pred_v13 — skipping exec grade")
        return pitchers, pit_types

    scored = pl.read_parquet(scored_path, columns=["game_pk", "pitcher_mlbid", "pitch_tag", "pred_v13"])

    # Pitcher-level: mean pred_v13 per (game_pk, pitcher_mlbid)
    p_pred = (
        scored.group_by(["game_pk", "pitcher_mlbid"])
        .agg(pl.col("pred_v13").mean().alias("pred_v13"))
    )
    if "game_pk" in pitchers.columns and "pitcher_mlbid" in pitchers.columns:
        pitchers = pitchers.join(p_pred, on=["game_pk", "pitcher_mlbid"], how="left")

    # Pitch-type level: mean pred_v13 per (game_pk, pitcher_mlbid, pitch_tag)
    pt_pred = (
        scored.group_by(["game_pk", "pitcher_mlbid", "pitch_tag"])
        .agg(pl.col("pred_v13").mean().alias("pred_v13"))
    )
    if "game_pk" in pit_types.columns and "pitcher_mlbid" in pit_types.columns and "pitch_tag" in pit_types.columns:
        pit_types = pit_types.join(pt_pred, on=["game_pk", "pitcher_mlbid", "pitch_tag"], how="left")

    print(f"  pred_v13 merged from {scored_path.name}")
    return pitchers, pit_types


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_chunk(df: pl.DataFrame, stem: str, season: int, chunk_dir: Path) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out = chunk_dir / f"{stem}.season_{season}.parquet"
    df.write_parquet(out)
    print(f"  Saved chunk: {out.name}  ({len(df):,} rows)")


def save_direct(df: pl.DataFrame, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.parquet"
    df.write_parquet(out)
    print(f"  Saved: {out.name}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build game-level aggregates.")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--scored-path", default=None,
                        help="Path to location_v13_scored.parquet (must include game_pk)")
    parser.add_argument("--min-season", type=int, default=None)
    parser.add_argument("--max-season", type=int, default=None)
    parser.add_argument("--chunk-by-season", action="store_true")
    parser.add_argument("--chunk-dir", default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    parquet_path = Path(args.parquet_path)
    out_dir      = Path(args.out_dir)
    chunk_dir    = Path(args.chunk_dir) if args.chunk_dir else CHUNK_DIR
    scored_path  = Path(args.scored_path) if args.scored_path else None

    t0 = perf_counter()
    print(f"Loading {parquet_path.name}...")
    df = pl.read_parquet(parquet_path)
    print(f"  {len(df):,} rows loaded  ({perf_counter()-t0:.1f}s)")

    if args.min_season:
        df = df.filter(pl.col("season") >= args.min_season)
    if args.max_season:
        df = df.filter(pl.col("season") <= args.max_season)

    # Filter pitchouts
    if "pitch_outcome" in df.columns:
        df = df.filter(pl.col("pitch_outcome") != "NIP")

    df = _add_game_type_group(df)

    seasons = sorted(df["season"].drop_nulls().unique().to_list())
    print(f"  Seasons: {seasons}")

    for season in seasons:
        t1 = perf_counter()
        print(f"\n--- Season {season} ---")
        sdf = df.filter(pl.col("season") == season)

        hitters   = build_hitter_gamelogs(sdf)
        pitchers  = build_pitcher_gamelogs(sdf)
        pit_types = build_pitch_type_gamelogs(sdf)

        if scored_path and scored_path.exists():
            # Filter scored to this season if it has a season column
            scored_schema = pl.read_parquet_schema(scored_path)
            if "season" in scored_schema:
                scored_season = pl.read_parquet(scored_path, columns=["season"]).filter(
                    pl.col("season") == season
                )
                if len(scored_season) > 0:
                    pitchers, pit_types = _merge_scored_pred_v13(pitchers, pit_types, scored_path)
            else:
                pitchers, pit_types = _merge_scored_pred_v13(pitchers, pit_types, scored_path)

        print(f"  hitter rows:     {len(hitters):,}")
        print(f"  pitcher rows:    {len(pitchers):,}")
        print(f"  pitch_type rows: {len(pit_types):,}")

        if args.chunk_by_season:
            save_chunk(hitters,   "hitter_gamelogs",     season, chunk_dir)
            save_chunk(pitchers,  "pitcher_gamelogs",    season, chunk_dir)
            save_chunk(pit_types, "pitch_type_gamelogs", season, chunk_dir)
        else:
            save_direct(hitters,   "hitter_gamelogs",     out_dir)
            save_direct(pitchers,  "pitcher_gamelogs",    out_dir)
            save_direct(pit_types, "pitch_type_gamelogs", out_dir)

        print(f"  Season {season} done  ({perf_counter()-t1:.1f}s)")

    print(f"\nTotal: {perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
