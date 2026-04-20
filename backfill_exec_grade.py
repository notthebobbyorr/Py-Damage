# -*- coding: utf-8 -*-
"""
Backfill Execution Grade (Location V13) for Historical Seasons

For each season in MIN_SEASON..MAX_SEASON this script:
  1. Scores each pitch with Location V13 using existing raw parquets from
     data/raw/_hist_seasons/pitch_data_{season}.parquet
  2. Writes per-season execution-grade chunks to _season_chunks/
  3. Regenerates gamelog chunks for that season (with pred_v13 attached)
  4. After all seasons are processed, stitches chunks into final output files
  5. Commits and pushes to GitHub

Usage:
    python backfill_exec_grade.py
    python backfill_exec_grade.py --min-season 2020 --max-season 2025
    python backfill_exec_grade.py --min-season 2023 --dry-run
    python backfill_exec_grade.py --no-push
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "output"
RAW_DIR = HERE / "data" / "raw"
HIST_DIR = RAW_DIR / "_hist_seasons"
CHUNK_DIR = DATA_DIR / "_season_chunks"

DEFAULT_MIN_SEASON = 2015
DEFAULT_MAX_SEASON = 2025

STITCHED_TABLES = [
    "location_v13_pitcher_season",
    "location_v13_pitcher_pitch",
    "pitcher_gamelogs",
    "pitch_type_gamelogs",
    "hitter_gamelogs",
]

# Gamelog tables written with pandas dtype optimisation so the app loads
# them efficiently (polars-written parquets expand ~20x in pandas memory).
PANDAS_OPTIMIZED_TABLES = {"pitcher_gamelogs", "pitch_type_gamelogs", "hitter_gamelogs"}


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=HERE)


def _write_optimized_parquet(df: pd.DataFrame, path: Path) -> None:
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / max(len(df[col]), 1) < 0.5:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    df.to_parquet(path, index=False)


def stitch(min_season: int, max_season: int) -> None:
    print(f"\nStitching chunks ({min_season}-{max_season}) -> {DATA_DIR}")
    for stem in STITCHED_TABLES:
        chunks = sorted(CHUNK_DIR.glob(f"{stem}.season_*.parquet"))
        if not chunks:
            print(f"  No chunks for {stem}, skipping.")
            continue
        out = DATA_DIR / f"{stem}.parquet"
        if stem in PANDAS_OPTIMIZED_TABLES:
            df = pl.concat(
                [pl.scan_parquet(p) for p in chunks], how="diagonal_relaxed"
            ).collect().to_pandas()
            _write_optimized_parquet(df, out)
        else:
            combined = pl.concat([pl.scan_parquet(p) for p in chunks], how="diagonal_relaxed")
            combined.sink_parquet(out)
        print(f"  {out.name}  ({len(chunks)} chunks)")


def main(
    min_season: int,
    max_season: int,
    dry_run: bool,
    no_push: bool,
) -> None:
    seasons = list(range(min_season, max_season + 1))
    print(f"Backfilling execution grade for seasons: {seasons}")

    for season in seasons:
        raw_path = HIST_DIR / f"pitch_data_{season}.parquet"
        if not raw_path.exists():
            print(f"\n[{season}] Raw parquet not found at {raw_path}, skipping.")
            continue

        # location_v13_apply.py defaults output_dir to data_path.parent, so for
        # historical data at _hist_seasons/ the scored parquet lands there too.
        scored_path = HIST_DIR / "location_v13_scored.parquet"
        print(f"\n{'='*60}\n[{season}] {raw_path.name}\n{'='*60}")

        # ── Score with Location V13 ────────────────────────────────────────
        run(
            [
                sys.executable, str(HERE / "pipeline" / "location_v13_apply.py"),
                "--data-path", str(raw_path),
                "--seasons", str(season),
                "--chunk-by-season",
                "--chunk-dir", str(CHUNK_DIR),
            ],
            dry_run=dry_run,
        )

        # ── Rebuild gamelogs for this season ──────────────────────────────
        gamelog_cmd = [
            sys.executable, str(HERE / "pipeline" / "build_gamelogs.py"),
            "--parquet-path", str(raw_path),
            "--min-season", str(season),
            "--max-season", str(season),
            "--chunk-by-season",
            "--chunk-dir", str(CHUNK_DIR),
            "--scored-path", str(scored_path),
        ]
        run(gamelog_cmd, dry_run=dry_run)

    # ── Stitch all chunks into final output files ──────────────────────────
    if not dry_run:
        stitch(min_season, max_season)

    # ── Commit and push ────────────────────────────────────────────────────
    if not no_push and not dry_run:
        run(["git", "add", str(DATA_DIR)])
        run(["git", "commit", "-m",
             f"Backfill execution grade for seasons {min_season}–{max_season}"])
        run(["git", "push"])
        print("\nBackfill complete and pushed.")
    else:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Backfill complete. Run without --no-push to commit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Location V13 execution grade.")
    parser.add_argument("--min-season", type=int, default=DEFAULT_MIN_SEASON)
    parser.add_argument("--max-season", type=int, default=DEFAULT_MAX_SEASON)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip git commit and push at the end.")
    args = parser.parse_args()
    main(
        min_season=args.min_season,
        max_season=args.max_season,
        dry_run=args.dry_run,
        no_push=args.no_push,
    )
