"""
blend_constants.py

Generates blended stability constants for the current season by combining:
  - Prior season's K (stable year-over-year, always kept as-is)
  - A weighted blend of prior-season and current-season mu (league-average ballast)

Blending is n_players weighted:
  - If n_current < min_players : use prior mu entirely (not enough current data)
  - If n_current >= transition_threshold * n_prior : use current mu fully
  - Otherwise : blended_mu = (n_prior * mu_prior + n_current * mu_current) / (n_prior + n_current)

Quantile/max stats are skipped (bootstrap-derived K not available mid-season).
Derived stats are skipped.
Baserunning constants are pooled (no season column) and are not modified.

Writes/replaces season=<target_season> rows in stability_constants.csv and .parquet.

Usage (called daily by run_daily_refresh.py before apply_regression_from_agg.py):
    python pipeline/blend_constants.py \\
        --parquet-path data/raw/pitch_data_2026.parquet \\
        --season 2026 --prior-season 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from polars.exceptions import ColumnNotFoundError

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR / "model_dev"))
from stability_constants import (  # noqa: E402
    grouped_rate,
    grouped_value,
    load_config,
    tag_pitch,
)

CONFIG_PATH = _REPO_DIR / "config" / "stability_config.yml"
CONSTANTS_CSV = _REPO_DIR / "config" / "stability_constants.csv"
CONSTANTS_PARQUET = _REPO_DIR / "config" / "stability_constants.parquet"

# Datasets processed from pitch-level data (baserunning is pooled, excluded here)
PITCH_DATASETS = ("hitters", "pitchers", "pitch_types")


def _dataset_cols(dataset: str) -> Dict[str, List[str]]:
    if dataset == "pitch_types":
        return {
            "group_cols": ["pitcher_mlbid", "season", "level_id", "pitch_tag"],
            "context_cols": ["level_id", "pitch_tag"],
        }
    if dataset == "pitchers":
        return {
            "group_cols": ["pitcher_mlbid", "season", "level_id"],
            "context_cols": ["level_id"],
        }
    # hitters
    return {
        "group_cols": ["batter_mlbid", "season", "level_id"],
        "context_cols": ["level_id"],
    }


def compute_current_mu(
    pitch: pl.DataFrame,
    config: Dict[str, Any],
    min_denom: int,
) -> pl.DataFrame:
    """
    For each (dataset, stat, context) compute mu and n_players from current pitch data.
    Skips quantile, max, and derived stats.
    Returns DataFrame with [dataset, stat, level_id, pitch_tag?, mu_current, n_players_current].
    """
    stats_cfg = config.get("stats", {})
    outputs: List[pl.DataFrame] = []

    for dataset in PITCH_DATASETS:
        cols = _dataset_cols(dataset)
        group_cols = cols["group_cols"]
        context_cols = cols["context_cols"]

        for stat in stats_cfg.get(dataset, []):
            kind = stat.get("kind")
            name = stat.get("name")

            if kind in ("derived", "quantile", "max"):
                continue

            # Some model-output columns (e.g. p_Swing_pct) are built during
            # aggregation and don't exist in the raw pitch parquet. Skip silently.
            try:
                if kind == "rate":
                    player = grouped_rate(pitch, stat, group_cols, min_denom)
                elif kind == "mean":
                    player = grouped_value(pitch, stat, group_cols, min_denom)
                else:
                    continue
            except (ColumnNotFoundError, Exception) as exc:
                if "unable to find column" in str(exc) or isinstance(exc, ColumnNotFoundError):
                    continue
                raise

            if player.is_empty():
                continue

            ctx = (
                player
                .group_by(context_cols)
                .agg([
                    pl.mean("stat").cast(pl.Float64).alias("mu_current"),
                    pl.len().cast(pl.Int64).alias("n_players_current"),
                ])
                .with_columns([
                    pl.lit(dataset).cast(pl.String).alias("dataset"),
                    pl.lit(name).cast(pl.String).alias("stat"),
                    pl.col("level_id").cast(pl.Int64),
                ])
            )
            if "pitch_tag" in ctx.columns:
                ctx = ctx.with_columns(pl.col("pitch_tag").cast(pl.String))
            outputs.append(ctx)

    if not outputs:
        return pl.DataFrame()

    return pl.concat(outputs, how="diagonal")


def build_blended_rows(
    prior: pl.DataFrame,
    current_mu: pl.DataFrame,
    target_season: int,
    min_players: int,
    transition_threshold: float,
) -> pl.DataFrame:
    """
    Left-join prior constants with current-season mu, apply blending logic,
    and return a DataFrame of target_season rows in stability_constants schema.
    K and derived variance columns are carried forward from prior season.
    """
    # Ensure pitch_tag exists on both sides for a clean join
    if "pitch_tag" not in prior.columns:
        prior = prior.with_columns(pl.lit(None).cast(pl.String).alias("pitch_tag"))
    if "pitch_tag" not in current_mu.columns:
        current_mu = current_mu.with_columns(pl.lit(None).cast(pl.String).alias("pitch_tag"))

    join_keys = ["dataset", "stat", "level_id", "pitch_tag"]

    join_kwargs: dict = {}
    try:
        import polars as _pl
        if int(_pl.__version__.split(".")[1]) >= 24:
            join_kwargs["nulls_equal"] = True
        else:
            join_kwargs["join_nulls"] = True
    except Exception:
        join_kwargs["join_nulls"] = True

    merged = prior.join(
        current_mu.select(join_keys + ["mu_current", "n_players_current"]),
        on=join_keys,
        how="left",
        **join_kwargs,
    )

    # n_prior as float for arithmetic
    n_prior = pl.col("n_players").cast(pl.Float64)
    n_current = pl.col("n_players_current").cast(pl.Float64)
    mu_prior = pl.col("mu")
    mu_current = pl.col("mu_current")

    blended = (n_prior * mu_prior + n_current * mu_current) / (n_prior + n_current)

    merged = merged.with_columns(
        pl.when(
            pl.col("mu_current").is_null()
            | (pl.col("n_players_current") < min_players)
        )
        .then(mu_prior)                                            # not enough 2026 data
        .when(n_current >= n_prior * transition_threshold)
        .then(mu_current)                                          # 2026 has enough, use fully
        .otherwise(blended)                                        # weighted blend
        .alias("mu_new")
    )

    # Log blend decisions
    rows = merged.select([
        "dataset", "stat", "level_id", "pitch_tag",
        "n_players", "n_players_current", "mu", "mu_current", "mu_new",
    ]).to_dicts()

    for r in rows:
        n_c = r["n_players_current"]
        n_p = r["n_players"]
        tag = f" [{r['pitch_tag']}]" if r["pitch_tag"] else ""
        ctx = f"{r['dataset']}.{r['stat']} level={r['level_id']}{tag}"
        if n_c is None or n_c < min_players:
            label = f"prior only (n_2026={n_c or 0} < {min_players})"
        elif n_p and n_c >= n_p * transition_threshold:
            label = f"current only (n_2026={n_c:.0f} >= {transition_threshold:.0%} of n_prior={n_p:.0f})"
        else:
            w = n_c / (n_p + n_c) if n_p else 1.0
            label = f"blended {w:.1%} current (n_2026={n_c:.0f}, n_prior={n_p:.0f})"
        print(f"  {ctx}: mu {r['mu']:.4f} -> {r['mu_new']:.4f}  [{label}]")

    # Build output columns matching stability_constants schema
    result = merged.with_columns([
        pl.lit(float(target_season)).alias("season"),
        pl.col("mu_new").alias("mu"),
        # n_players: report current season count where available, else prior
        pl.when(pl.col("n_players_current").is_not_null())
        .then(pl.col("n_players_current"))
        .otherwise(pl.col("n_players"))
        .cast(pl.Int64)
        .alias("n_players"),
        # Variance columns are not recomputed; null them for the blended rows
        pl.lit(None).cast(pl.Float64).alias("var_obs"),
        pl.lit(None).cast(pl.Float64).alias("var_within"),
        pl.lit(None).cast(pl.Float64).alias("var_true"),
        pl.lit(None).cast(pl.Float64).alias("var_between"),
        # K, n_50, n_70, n_80 carried forward from prior (stable year-over-year)
    ])

    desired_cols = [
        "dataset", "stat", "kind", "season", "level_id", "pitch_tag",
        "n_players", "mu", "var_obs", "var_within", "var_true", "var_between",
        "K", "n_50", "n_70", "n_80",
    ]
    existing = [c for c in desired_cols if c in result.columns]
    return result.select(existing)


def update_constants(
    blended: pl.DataFrame,
    target_season: int,
    csv_path: Path,
    parquet_path: Path,
) -> None:
    """Drop existing target_season rows and append blended rows to both files."""
    existing = pl.read_csv(csv_path)
    kept = existing.filter(pl.col("season").is_null() | (pl.col("season") != float(target_season)))

    # Align blended dtypes to match the existing CSV schema to avoid concat errors
    casts = []
    for col_name in blended.columns:
        if col_name in existing.columns and blended[col_name].dtype != existing[col_name].dtype:
            casts.append(pl.col(col_name).cast(existing[col_name].dtype, strict=False))
    if casts:
        blended = blended.with_columns(casts)

    updated = pl.concat([kept, blended], how="diagonal")

    updated.write_csv(csv_path)
    updated.write_parquet(parquet_path)
    n_new = len(blended)
    n_total = len(updated)
    print(f"\nWrote {n_new} season={target_season} rows ({n_total} total) to {csv_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend prior-season and current-season stability constants.")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        required=True,
        help="Path to current-season pitch-level parquet (e.g. data/raw/pitch_data_2026.parquet).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Target season to generate constants for (default: 2026).",
    )
    parser.add_argument(
        "--prior-season",
        type=int,
        default=None,
        help="Prior season to use as base (default: season - 1).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to stability_config.yml.",
    )
    parser.add_argument(
        "--constants",
        type=Path,
        default=CONSTANTS_CSV,
        help="Path to stability_constants.csv.",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=10,
        help="Minimum qualifying players in current season to include in blend (default: 10).",
    )
    parser.add_argument(
        "--min-denom",
        type=int,
        default=20,
        help="Minimum denominator/sample_n for player-level stats (default: 20).",
    )
    parser.add_argument(
        "--transition-threshold",
        type=float,
        default=0.80,
        help="Fraction of prior n_players at which current season mu is used fully (default: 0.80).",
    )
    args = parser.parse_args()

    prior_season = args.prior_season if args.prior_season is not None else args.season - 1
    parquet_out = args.constants.with_suffix(".parquet")

    print(f"blend_constants: season={args.season}, prior={prior_season}, "
          f"transition_threshold={args.transition_threshold:.0%}, min_players={args.min_players}")

    # Load config and pitch data
    config = load_config(args.config)
    pitch = pl.read_parquet(args.parquet_path)
    pitch = tag_pitch(pitch)
    print(f"Loaded {len(pitch):,} pitch rows from {args.parquet_path.name}")

    # Load prior season constants
    all_constants = pl.read_csv(args.constants)
    prior = all_constants.filter(pl.col("season") == float(prior_season))
    if prior.is_empty():
        print(f"ERROR: No constants found for prior season {prior_season}. Aborting.")
        sys.exit(1)
    print(f"Loaded {len(prior)} prior-season ({prior_season}) constant rows\n")

    # Compute current-season mu
    print("Computing current-season mu from pitch data...")
    current_mu = compute_current_mu(pitch, config, args.min_denom)
    print(f"Computed {len(current_mu)} current-season mu values\n")

    # Blend
    print("Blending mu values:")
    blended = build_blended_rows(
        prior,
        current_mu,
        target_season=args.season,
        min_players=args.min_players,
        transition_threshold=args.transition_threshold,
    )

    # Write
    update_constants(blended, args.season, args.constants, parquet_out)


if __name__ == "__main__":
    main()
