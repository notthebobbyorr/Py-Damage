# -*- coding: utf-8 -*-
"""
Daily Pipeline Refresh Orchestrator

Runs the full incremental data pipeline:
  1. Read last pull date from data/logs/last_pull_date.txt
  2. Pull new pitch rows from DB (start_date to end_date)
  3. Accumulate new rows into data/raw/pitch_data_{season}.parquet
  4. Re-aggregate current season chunk
  5. Build p(damage) and p(swstr) probability source tables
  6. Merge probability tables into aggregated output files
  7. Stitch all season chunks into final output files
  8. Apply regression
  9. Update last_pull_date.txt

Usage:
    python run_daily_refresh.py
    python run_daily_refresh.py --season 2026 --dry-run
    python run_daily_refresh.py --game-types R S F D L W
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "output"
RAW_DIR = HERE / "data" / "raw"
LOGS_DIR = HERE / "data" / "logs"
CHUNK_DIR = DATA_DIR / "_season_chunks"
LAST_PULL_DATE_FILE = LOGS_DIR / "last_pull_date.txt"
REQUIREMENTS_FILE = HERE / "requirements.txt"

DEFAULT_GAME_TYPES = ["R", "S", "F", "D", "L", "W"]

# Output table stem names (without .parquet) that are stitched from season chunks
STITCHED_TABLES = [
    "pitcher_stuff_new",
    "new_pitch_types",
    "new_team_damage",
    "new_team_stuff",
    "league_pitch_types",
    "hitter_splits",
    "pitcher_splits",
    "pitch_types_splits",
    "hitter_pctiles",
    "pitcher_pctiles",
    "pitch_types_pctiles",
    "new_hitting_lg_avg",
    "new_lg_stuff",
    "park_data",
    "baserunning",
    "pitcher_baserunning",
    "location_v13_pitcher_season",
    "location_v13_pitcher_pitch",
    "hitter_gamelogs",
    "pitcher_gamelogs",
    "pitch_type_gamelogs",
]


def read_last_pull_date() -> date:
    if not LAST_PULL_DATE_FILE.exists():
        raise FileNotFoundError(
            f"Last pull date file not found: {LAST_PULL_DATE_FILE}\n"
            "Create it with a date in YYYY-MM-DD format (e.g. 2025-09-28)."
        )
    text = LAST_PULL_DATE_FILE.read_text(encoding="utf-8").strip()
    return date.fromisoformat(text)


def write_last_pull_date(d: date) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PULL_DATE_FILE.write_text(d.isoformat(), encoding="utf-8")
    print(f"Updated last_pull_date.txt -> {d.isoformat()}")


def run(cmd: list[str], dry_run: bool = False, cwd: Path | None = None) -> None:
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd, check=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def _stamp_requirements(end_date: date) -> None:
    """Update the last-refresh comment in requirements.txt.

    Streamlit Cloud triggers a full process restart (clearing all st.cache_resource
    entries) when requirements.txt changes. Stamping it daily ensures fresh data
    is loaded after each push rather than a hot-reload that preserves the cache.
    """
    content = REQUIREMENTS_FILE.read_text(encoding="utf-8")
    lines = [l for l in content.splitlines() if not l.startswith("# last-refresh:")]
    lines.append(f"# last-refresh: {end_date.isoformat()}")
    REQUIREMENTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Stamped requirements.txt -> last-refresh: {end_date.isoformat()}")


def git_push(end_date: date, dry_run: bool = False) -> None:
    """Stage output parquet files, commit, and push to GitHub.

    If the tip of main is already a 'daily data update' commit, amend it
    in-place rather than adding a new commit. This keeps exactly one data
    commit on top of the code history, so the clone size stays constant.
    """
    commit_msg = "daily data update"
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Git push: staging data/output/, committing, and pushing.")
    if not dry_run:
        _stamp_requirements(end_date)
    run(["git", "add", str(DATA_DIR), str(REQUIREMENTS_FILE)], dry_run=dry_run, cwd=HERE)

    # Amend if the last commit is a data update, otherwise create a new commit.
    amend = False
    if not dry_run:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"], capture_output=True, text=True, cwd=HERE
        )
        amend = result.stdout.strip() == commit_msg

    if amend:
        run(["git", "commit", "--amend", "-m", commit_msg], dry_run=dry_run, cwd=HERE)
        run(["git", "push", "--force", "origin", "main"], dry_run=dry_run, cwd=HERE)
    else:
        run(["git", "commit", "-m", commit_msg], dry_run=dry_run, cwd=HERE)
        run(["git", "push"], dry_run=dry_run, cwd=HERE)


def accumulate_pitch_data(incremental_path: Path, accumulator_path: Path) -> None:
    """Append incremental rows to season accumulator parquet, deduplicating on (pa_id, pitch_of_ab)."""
    print(f"Accumulating {incremental_path} -> {accumulator_path}")
    incremental = pl.read_parquet(incremental_path)
    print(f"  Incremental rows: {len(incremental):,}")

    existing_rows = 0
    if accumulator_path.exists():
        try:
            existing = pl.read_parquet(accumulator_path)
            existing_rows = len(existing)
            print(f"  Existing rows: {existing_rows:,}")
            combined = pl.concat([existing, incremental], how="diagonal_relaxed")
        except Exception as exc:
            corrupt_path = accumulator_path.with_suffix(".parquet.corrupt")
            accumulator_path.rename(corrupt_path)
            print(
                f"  WARNING: accumulator file is corrupt ({exc}).\n"
                f"  Moved corrupt file to {corrupt_path.name}.\n"
                f"  Falling back to incremental-only — re-run a full backfill to restore historical data."
            )
            combined = incremental
    else:
        print("  No existing accumulator found — using incremental as initial file.")
        combined = incremental

    dedup_keys = [c for c in ["pa_id", "pitch_of_ab"] if c in combined.columns]
    if dedup_keys:
        before = len(combined)
        combined = combined.unique(subset=dedup_keys, keep="last")
        dropped = before - len(combined)
        if dropped:
            print(f"  Removed {dropped:,} duplicate rows on {dedup_keys}.")

    new_rows = len(combined)
    if new_rows < existing_rows:
        raise RuntimeError(
            f"Accumulator sanity check failed: new row count ({new_rows:,}) is less than "
            f"existing row count ({existing_rows:,}). Aborting to prevent data loss."
        )

    # Write atomically: write to a temp file then rename to avoid partial writes
    accumulator_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = accumulator_path.with_suffix(".parquet.tmp")
    combined.write_parquet(tmp_path)
    tmp_path.replace(accumulator_path)
    print(f"  Accumulator now has {new_rows:,} rows.")


def stitch_season_chunks(min_season: int, current_season: int) -> None:
    """Combine per-season chunks into final output files in DATA_DIR."""
    print(f"\nStitching season chunks ({min_season}-{current_season}) -> {DATA_DIR}")

    # Handle the hitters table separately — its chunk name contains the season range
    hitter_chunks = sorted(CHUNK_DIR.glob("damage_pos_*.season_*.parquet"))
    if hitter_chunks:
        combined = pl.concat(
            [pl.scan_parquet(p) for p in hitter_chunks], how="diagonal_relaxed"
        )
        out_path = DATA_DIR / f"damage_pos_{min_season}_{current_season}.parquet"
        combined.sink_parquet(out_path)
        print(f"  Wrote {out_path.name} from {len(hitter_chunks)} chunks")

    # All other tables have consistent stem names
    for stem in STITCHED_TABLES:
        chunks = sorted(CHUNK_DIR.glob(f"{stem}.season_*.parquet"))
        if not chunks:
            print(f"  No chunks found for {stem}, skipping.")
            continue
        combined = pl.concat(
            [pl.scan_parquet(p) for p in chunks], how="diagonal_relaxed"
        )
        out_path = DATA_DIR / f"{stem}.parquet"
        combined.sink_parquet(out_path)
        print(f"  Wrote {out_path.name} from {len(chunks)} chunks")


def main(
    season: int,
    min_season: int,
    game_types: list[str],
    dry_run: bool,
    no_push: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    # Ensure we are on main before doing anything — prevents committing to a feature branch
    run(["git", "checkout", "main"], dry_run=dry_run, cwd=HERE)

    last_pull_date = read_last_pull_date()
    if start_date is None:
        start_date = last_pull_date + timedelta(days=1)
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    print(f"Last pull date : {last_pull_date}")
    print(f"Pull range     : {start_date} -> {end_date}")
    print(f"Current season : {season}")
    print(f"Game types     : {game_types}")

    if start_date > end_date:
        print("No new dates to pull. Pipeline complete.")
        return

    # ── Step 1: Incremental data pull ──────────────────────────────────────
    incremental_path = RAW_DIR / f"pitch_data_{season}_incremental.parquet"
    run(
        [
            sys.executable, str(HERE / "pipeline" / "data_pull.py"),
            "--min-season", str(season),
            "--max-season", str(season),
            "--start-date", start_date.isoformat(),
            "--end-date", end_date.isoformat(),
            "--game-types", *game_types,
            "--out-file", str(incremental_path),
        ],
        dry_run=dry_run,
    )

    # ── Step 2: Accumulate into season parquet ─────────────────────────────
    accumulator_path = RAW_DIR / f"pitch_data_{season}.parquet"
    if not dry_run:
        accumulate_pitch_data(incremental_path, accumulator_path)

    # ── Step 2b: Score pitch execution (Location V13) ─────────────────────
    run(
        [
            sys.executable, str(HERE / "pipeline" / "location_v13_apply.py"),
            "--data-path", str(accumulator_path),
            "--seasons", str(season),
            "--chunk-by-season",
            "--chunk-dir", str(CHUNK_DIR),
        ],
        dry_run=dry_run,
    )

    # ── Step 3: Re-aggregate current season chunk ──────────────────────────
    run(
        [
            sys.executable, str(HERE / "pipeline" / "data_aggregate.py"),
            "--parquet-path", str(accumulator_path),
            "--min-season", str(season),
            "--max-season", str(season),
            "--chunk-by-season",
            "--chunk-dir", str(CHUNK_DIR),
            "--out-dir", str(DATA_DIR),
        ],
        dry_run=dry_run,
    )

    # ── Step 3b: Build game-by-game gamelogs ──────────────────────────────
    _scored_path = RAW_DIR / "location_v13_scored.parquet"
    _gamelog_cmd = [
        sys.executable, str(HERE / "pipeline" / "build_gamelogs.py"),
        "--parquet-path", str(accumulator_path),
        "--min-season", str(season),
        "--max-season", str(season),
        "--chunk-by-season",
        "--chunk-dir", str(CHUNK_DIR),
    ]
    if _scored_path.exists():
        _gamelog_cmd += ["--scored-path", str(_scored_path)]
    run(_gamelog_cmd, dry_run=dry_run)

    # ── Step 4a: Build p(damage) source tables ─────────────────────────────
    run(
        [
            sys.executable, str(HERE / "pipeline" / "build_p_damage_sources.py"),
            "--parquet-path", str(accumulator_path),
        ],
        dry_run=dry_run,
    )

    # ── Step 4b: Build p(swstr) source tables ──────────────────────────────
    run(
        [
            sys.executable, str(HERE / "pipeline" / "build_p_swstr_sources.py"),
            "--parquet-path", str(accumulator_path),
        ],
        dry_run=dry_run,
    )

    # ── Step 4c: Build hitter p(swstr/swing) source tables ─────────────────
    run(
        [
            sys.executable, str(HERE / "pipeline" / "build_p_swstr_hitter_sources.py"),
            "--parquet-path", str(accumulator_path),
        ],
        dry_run=dry_run,
    )

    # ── Step 4d: Build hitter p(damage) source tables ──────────────────────
    run(
        [
            sys.executable, str(HERE / "pipeline" / "build_p_damage_hitter_sources.py"),
            "--parquet-path", str(accumulator_path),
        ],
        dry_run=dry_run,
    )

    # ── Step 5: Stitch all season chunks into final output ─────────────────
    if not dry_run:
        stitch_season_chunks(min_season, season)

    # ── Step 6a: Merge p(damage) into aggregated files ─────────────────────
    run(
        [sys.executable, str(HERE / "pipeline" / "merge_p_damage_into_sources.py")],
        dry_run=dry_run,
    )

    # ── Step 6b: Merge p(swstr) into aggregated files ──────────────────────
    run(
        [sys.executable, str(HERE / "pipeline" / "merge_p_swstr_into_sources.py")],
        dry_run=dry_run,
    )

    # ── Step 6c: Merge hitter model outputs into damage_pos ────────────────
    run(
        [sys.executable, str(HERE / "pipeline" / "merge_model_outputs_into_damage_pos.py")],
        dry_run=dry_run,
    )

    # ── Step 6d: Blend stability constants for current season ──────────────
    run(
        [
            sys.executable, str(HERE / "pipeline" / "blend_constants.py"),
            "--parquet-path", str(RAW_DIR / f"pitch_data_{season}.parquet"),
            "--season", str(season),
        ],
        dry_run=dry_run,
    )

    # ── Step 7: Apply regression ───────────────────────────────────────────
    hitters_path = DATA_DIR / f"damage_pos_{min_season}_{season}.parquet"
    run(
        [
            sys.executable, str(HERE / "pipeline" / "apply_regression_from_agg.py"),
            "--hitters", str(hitters_path),
            "--baserunning", str(DATA_DIR / "baserunning.parquet"),
            "--pitcher-baserunning", str(DATA_DIR / "pitcher_baserunning.parquet"),
        ],
        dry_run=dry_run,
    )

    # ── Step 8: Update last pull date ──────────────────────────────────────
    if not dry_run:
        write_last_pull_date(end_date)
        print(f"\nPipeline complete. Data refreshed through {end_date}.")
    else:
        print(f"\n[DRY RUN] Pipeline complete. Would update last_pull_date to {end_date}.")

    # ── Step 9: Push to GitHub ─────────────────────────────────────────────
    if not no_push:
        git_push(end_date, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily data refresh pipeline.")
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Current season to pull and re-aggregate (default: 2026).",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=2015,
        help="Earliest season in the historical chunk archive (default: 2015).",
    )
    parser.add_argument(
        "--game-types",
        type=str,
        nargs="+",
        default=DEFAULT_GAME_TYPES,
        help="Game types to include in pull. Default: R S F D L W.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without executing them.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip the automatic git push to GitHub at the end of the pipeline.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Override pull start date (YYYY-MM-DD). Default: last_pull_date + 1 day.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Override pull end date (YYYY-MM-DD). Default: today - 1 day.",
    )
    args = parser.parse_args()
    main(
        season=args.season,
        min_season=args.min_season,
        game_types=args.game_types,
        dry_run=args.dry_run,
        no_push=args.no_push,
        start_date=args.start_date,
        end_date=args.end_date,
    )
