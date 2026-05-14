"""One-time backfill: re-aggregate historical season chunks with the new column set.

Run this once after merging the batch-work PR. The daily refresh only ever
re-aggregates the current season's chunk, so historical chunks need a one-time
catch-up to pick up:

- z_angle_release / x_angle_release on pitch_types          (all seasons 2015+)
- team_hitter_splits / team_pitcher_splits chunks           (all seasons)
- fast_swing_pct + attack_direction + intercept_x/y_inches  (Statcast 2023+)
- arm_angle now properly cast from String to Float64        (2020, 2024, 2025)
- bat_speed / swing_length / swing_path_tilt / attack_angle  (was silently null
  in production due to String dtype; now properly cast)

After running this, the next daily refresh's stitch step produces final
data/output/*.parquet with the new columns for every season where the source
data has them.

Usage:
    python pipeline/backfill_aggregates.py
    python pipeline/backfill_aggregates.py --min-season 2020 --max-season 2025
    python pipeline/backfill_aggregates.py --dry-run

Expects historical pitch data at data/raw/_hist_seasons/pitch_data_YYYY.parquet.
Each season takes ~3-5 minutes and peaks at ~3-4 GB RAM.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_RAW_HIST_DIR = HERE / "data" / "raw" / "_hist_seasons"
DEFAULT_DATA_DIR = HERE / "data" / "output"


def run_season(
    year: int,
    raw_hist_dir: Path,
    chunk_dir: Path,
    data_dir: Path,
    dry_run: bool,
) -> None:
    source = raw_hist_dir / f"pitch_data_{year}.parquet"
    if not source.exists():
        print(f"  SKIP {year}: source not found at {source}")
        return
    cmd = [
        sys.executable,
        str(HERE / "pipeline" / "data_aggregate.py"),
        "--parquet-path", str(source),
        "--min-season", str(year),
        "--max-season", str(year),
        "--chunk-by-season",
        "--chunk-dir", str(chunk_dir),
        "--out-dir", str(data_dir),
    ]
    print(f"\n=== {year} ===")
    print(" ".join(cmd))
    if dry_run:
        return
    t0 = time.time()
    subprocess.check_call(cmd)
    print(f"  {year} complete in {time.time() - t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-season",
        type=int,
        default=2015,
        help="First season to backfill (default: 2015).",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=2025,
        help="Last historical season to backfill (default: 2025). Do not include "
        "the current season — the daily refresh handles that.",
    )
    parser.add_argument(
        "--raw-hist-dir",
        type=Path,
        default=DEFAULT_RAW_HIST_DIR,
        help="Override the historical pitch data directory (default: data/raw/_hist_seasons).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Override the output directory (default: data/output).",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Override the per-season chunk directory (default: <out-dir>/_season_chunks).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    raw_hist_dir = args.raw_hist_dir.resolve()
    data_dir = args.out_dir.resolve()
    chunk_dir = (args.chunk_dir or (data_dir / "_season_chunks")).resolve()

    if not raw_hist_dir.exists():
        raise SystemExit(f"Historical pitch data directory not found: {raw_hist_dir}")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    seasons = list(range(args.min_season, args.max_season + 1))
    print(f"Backfill plan: {len(seasons)} seasons ({args.min_season}-{args.max_season})")
    print(f"  Source:    {raw_hist_dir}")
    print(f"  Chunk dir: {chunk_dir}")
    print(f"  Output:    {data_dir}")
    print(f"  Estimated wall time: ~{len(seasons) * 4} minutes")
    print()

    t_start = time.time()
    for year in seasons:
        run_season(year, raw_hist_dir, chunk_dir, data_dir, dry_run=args.dry_run)
    elapsed = time.time() - t_start

    print(f"\nBackfill complete in {elapsed / 60:.1f} minutes.")
    print(
        "Next step: run `python run_daily_refresh.py` (or wait for the next "
        "scheduled run) to stitch updated chunks into final data/output/*.parquet."
    )


if __name__ == "__main__":
    main()
