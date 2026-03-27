"""
One-time backfill script: add p(damage), p(swstr), and p(swing) model outputs
to 2015-2025 historical pitcher, pitch-type, and hitter aggregated files.

Pitcher / pitch-type pipeline (Steps 1-5):
  1. Build pitcher-level p(damage) source table from historical pitch data
  2. Build pitcher-level p(swstr) + p(swing) source table
  3. Combine historical results with existing current-season source tables
  4. Merge combined source tables into stitched output files
  5. Apply regression — pitchers and pitch_types

Hitter pipeline (Steps 6-8):
  6. Build batter-level p(damage) and p(swstr/swing) source tables
  7. Merge batter source tables into damage_pos hitter file
  8. Apply regression — hitters

  9. Clean up temporary files

Usage:
    python backfill_model_outputs.py
    python backfill_model_outputs.py --hist-parquet /path/to/custom.parquet
    python backfill_model_outputs.py --skip-build      # re-use existing _hist temp files
    python backfill_model_outputs.py --skip-hitters    # run pitcher pipeline only
    python backfill_model_outputs.py --skip-pitchers   # run hitter pipeline only
    python backfill_model_outputs.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
DATA_DIR = HERE / "data" / "output"

DEFAULT_HIST_PARQUET = Path(r"C:\Users\orrro\Documents\Baseball_Env\pitch_data_2015_2025_all_levels.parquet")

# ── Temporary output paths for historical build results ────────────────────
HIST_PITCHER_DAMAGE       = RAW_DIR / "pitcher_p_damage_hist.parquet"
HIST_PITCH_TYPES_DAMAGE   = RAW_DIR / "pitch_types_p_damage_hist.parquet"
HIST_PITCHER_SWSTR        = RAW_DIR / "pitcher_p_swstr_hist.parquet"
HIST_PITCH_TYPES_SWSTR    = RAW_DIR / "pitch_types_p_swstr_hist.parquet"
HIST_HITTER_SWSTR         = RAW_DIR / "hitter_p_swstr_hist.parquet"
HIST_HITTER_DAMAGE        = RAW_DIR / "hitter_p_damage_hist.parquet"

# ── Final combined source paths (used by merge scripts) ────────────────────
FINAL_PITCHER_DAMAGE      = RAW_DIR / "pitcher_p_damage.parquet"
FINAL_PITCH_TYPES_DAMAGE  = RAW_DIR / "pitch_types_p_damage.parquet"
FINAL_PITCHER_SWSTR       = RAW_DIR / "pitcher_p_swstr.parquet"
FINAL_PITCH_TYPES_SWSTR   = RAW_DIR / "pitch_types_p_swstr.parquet"
FINAL_HITTER_SWSTR        = RAW_DIR / "hitter_p_swstr.parquet"
FINAL_HITTER_DAMAGE       = RAW_DIR / "hitter_p_damage.parquet"


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def combine_sources(hist_path: Path, current_path: Path, dry_run: bool) -> None:
    """Concatenate historical and current-season source parquets, write to current_path."""
    print(f"\nCombining {hist_path.name} + {current_path.name} -> {current_path.name}")
    if dry_run:
        print("  [DRY RUN] skipping")
        return
    hist = pl.read_parquet(hist_path)
    print(f"  Historical rows: {len(hist):,}")
    if current_path.exists():
        cur = pl.read_parquet(current_path)
        print(f"  Current (2026) rows: {len(cur):,}")
        combined = pl.concat([hist, cur], how="diagonal_relaxed")
    else:
        print("  No current file found — using historical only.")
        combined = hist
    combined.write_parquet(current_path)
    print(f"  Combined rows written: {len(combined):,}")


def find_latest_damage_pos() -> Path | None:
    candidates = sorted(DATA_DIR.glob("damage_pos_2015_[0-9][0-9][0-9][0-9].parquet"))
    return candidates[-1] if candidates else None


def cleanup(paths: list[Path], dry_run: bool) -> None:
    for p in paths:
        if dry_run:
            print(f"  [DRY RUN] would delete {p}")
        elif p.exists():
            p.unlink()
            print(f"  Deleted {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill p(damage)/p(swstr)/p(swing) for pitchers and hitters 2015-2025."
    )
    parser.add_argument(
        "--hist-parquet", type=Path, default=DEFAULT_HIST_PARQUET,
        help="Path to historical pitch-level parquet.",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip build steps; re-use existing _hist temp files.",
    )
    parser.add_argument(
        "--skip-pitchers", action="store_true",
        help="Skip pitcher/pitch-type pipeline (steps 1-5).",
    )
    parser.add_argument(
        "--skip-hitters", action="store_true",
        help="Skip hitter pipeline (steps 6-8).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    if not args.skip_build and not args.hist_parquet.exists():
        raise FileNotFoundError(f"Historical parquet not found: {args.hist_parquet}")

    # ──────────────────────────────────────────────────────────────────────
    # PITCHER / PITCH-TYPE PIPELINE
    # ──────────────────────────────────────────────────────────────────────
    if not args.skip_pitchers:
        # Step 1: Build pitcher-level p(damage)
        if not args.skip_build:
            run(
                [
                    sys.executable, str(HERE / "build_p_damage_sources.py"),
                    "--parquet-path", str(args.hist_parquet),
                    "--pitcher-output", str(HIST_PITCHER_DAMAGE),
                    "--pitch-type-output", str(HIST_PITCH_TYPES_DAMAGE),
                ],
                dry_run=args.dry_run,
            )
        else:
            print("Skipping build_p_damage_sources (--skip-build).")

        # Step 2: Build pitcher-level p(swstr) + p(swing)
        if not args.skip_build:
            run(
                [
                    sys.executable, str(HERE / "build_p_swstr_sources.py"),
                    "--parquet-path", str(args.hist_parquet),
                    "--pitcher-output", str(HIST_PITCHER_SWSTR),
                    "--pitch-type-output", str(HIST_PITCH_TYPES_SWSTR),
                ],
                dry_run=args.dry_run,
            )
        else:
            print("Skipping build_p_swstr_sources (--skip-build).")

        # Step 3: Combine historical + current-season into final source files
        combine_sources(HIST_PITCHER_DAMAGE, FINAL_PITCHER_DAMAGE, args.dry_run)
        combine_sources(HIST_PITCH_TYPES_DAMAGE, FINAL_PITCH_TYPES_DAMAGE, args.dry_run)
        combine_sources(HIST_PITCHER_SWSTR, FINAL_PITCHER_SWSTR, args.dry_run)
        combine_sources(HIST_PITCH_TYPES_SWSTR, FINAL_PITCH_TYPES_SWSTR, args.dry_run)

        # Step 4: Merge into stitched pitcher/pitch-type output files
        run([sys.executable, str(HERE / "merge_p_damage_into_sources.py")], dry_run=args.dry_run)
        run([sys.executable, str(HERE / "merge_p_swstr_into_sources.py")], dry_run=args.dry_run)

        # Step 5: Apply regression for pitchers and pitch_types
        run(
            [
                sys.executable, str(HERE / "apply_regression_from_agg.py"),
                "--pitchers",    str(DATA_DIR / "pitcher_stuff_new.parquet"),
                "--pitch-types", str(DATA_DIR / "new_pitch_types.parquet"),
            ],
            dry_run=args.dry_run,
        )
    else:
        print("Skipping pitcher/pitch-type pipeline (--skip-pitchers).")

    # ──────────────────────────────────────────────────────────────────────
    # HITTER PIPELINE
    # ──────────────────────────────────────────────────────────────────────
    if not args.skip_hitters:
        # Step 6a: Build batter-level p(swstr) + p(swing)
        if not args.skip_build:
            run(
                [
                    sys.executable, str(HERE / "build_p_swstr_hitter_sources.py"),
                    "--parquet-path", str(args.hist_parquet),
                    "--batter-output", str(HIST_HITTER_SWSTR),
                ],
                dry_run=args.dry_run,
            )
        else:
            print("Skipping build_p_swstr_hitter_sources (--skip-build).")

        # Step 6b: Build batter-level p(damage)
        if not args.skip_build:
            run(
                [
                    sys.executable, str(HERE / "build_p_damage_hitter_sources.py"),
                    "--parquet-path", str(args.hist_parquet),
                    "--batter-output", str(HIST_HITTER_DAMAGE),
                ],
                dry_run=args.dry_run,
            )
        else:
            print("Skipping build_p_damage_hitter_sources (--skip-build).")

        # Combine historical hitter sources with current-season (if present)
        combine_sources(HIST_HITTER_SWSTR, FINAL_HITTER_SWSTR, args.dry_run)
        combine_sources(HIST_HITTER_DAMAGE, FINAL_HITTER_DAMAGE, args.dry_run)

        # Step 7: Merge batter source tables into damage_pos
        run([sys.executable, str(HERE / "merge_model_outputs_into_damage_pos.py")], dry_run=args.dry_run)

        # Step 8: Apply regression for hitters
        hitters_path = find_latest_damage_pos()
        if hitters_path is None and not args.dry_run:
            print("Warning: no damage_pos_2015_YYYY.parquet found — skipping hitter regression.")
        else:
            run(
                [
                    sys.executable, str(HERE / "apply_regression_from_agg.py"),
                    "--hitters", str(hitters_path or DATA_DIR / "damage_pos_2015_2025.parquet"),
                ],
                dry_run=args.dry_run,
            )
    else:
        print("Skipping hitter pipeline (--skip-hitters).")

    # ──────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────────────
    print("\nCleaning up temporary files...")
    cleanup(
        [
            HIST_PITCHER_DAMAGE, HIST_PITCH_TYPES_DAMAGE,
            HIST_PITCHER_SWSTR,  HIST_PITCH_TYPES_SWSTR,
            HIST_HITTER_SWSTR,   HIST_HITTER_DAMAGE,
        ],
        dry_run=args.dry_run,
    )

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
