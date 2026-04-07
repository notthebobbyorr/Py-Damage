# -*- coding: utf-8 -*-
"""
Backfill Baserunning Stats

Processes all historical season parquets and the current season to produce
a combined baserunning.parquet covering all available seasons.

Usage:
    python pipeline/backfill_baserunning.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import polars as pl

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_aggregate import build_baserunning, _add_game_type_group

HIST_DIR = _REPO_DIR / "data" / "raw" / "_hist_seasons"
RAW_DIR  = _REPO_DIR / "data" / "raw"
OUT_DIR  = _REPO_DIR / "data" / "output"

SEASON_FILES = sorted(HIST_DIR.glob("pitch_data_*.parquet")) + sorted(
    RAW_DIR.glob("pitch_data_[0-9][0-9][0-9][0-9].parquet")
)


def main() -> None:
    print(f"Found {len(SEASON_FILES)} season files to process:")
    for f in SEASON_FILES:
        print(f"  {f.name}")
    print()

    frames: list[pl.DataFrame] = []
    total_start = perf_counter()

    for path in SEASON_FILES:
        season_start = perf_counter()
        print(f"-- {path.name} ------------------------------")
        df = pl.read_parquet(path)
        df = _add_game_type_group(df)
        result = build_baserunning(df)
        elapsed = perf_counter() - season_start
        print(f"   {len(result):,} rows | {elapsed:.1f}s")
        frames.append(result)
        print()

    print("Concatenating all seasons...")
    combined = pl.concat(frames)
    print(f"Combined shape: {combined.shape}")

    out_path = OUT_DIR / "baserunning.parquet"
    combined.write_parquet(out_path)
    print(f"Wrote {len(combined):,} rows to {out_path}")
    print(f"Total time: {perf_counter() - total_start:.1f}s")

    # Summary by season and level
    print("\nSummary (Regular Season only):")
    summary = (
        combined.filter(pl.col("game_type_group") == "Regular Season")
        .group_by(["season", "level_id"])
        .agg([
            pl.len().alias("players"),
            pl.col("SBO").sum(),
            pl.col("SB").sum(),
            pl.col("CS").sum(),
        ])
        .sort(["season", "level_id"])
    )
    print(summary.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
