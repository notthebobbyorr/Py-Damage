"""Fit direct contact-over-expected priors from regular-season swing residuals."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

REPO_DIR = Path(__file__).resolve().parent.parent


def contact_players(pitch: pl.LazyFrame) -> pl.DataFrame:
    """One residual mean/variance per hitter-season-level, in percentage points."""
    return (
        pitch.filter(
            (pl.col("game_type") == "R") & (pl.col("swing") == 1)
            & pl.col("pred_whiff_loc").is_finite()
            & pl.col("whiff").is_in([0, 1])
        )
        .with_columns(
            (100.0 * (pl.col("pred_whiff_loc") - pl.col("whiff"))).alias("coe")
        )
        .group_by(["batter_mlbid", "season", "level_id"])
        .agg(pl.col("coe").mean().alias("stat"),
             pl.col("coe").var(ddof=1).alias("var"), pl.len().alias("n"))
        .collect(engine="streaming")
    )


def contact_constants(
    player: pl.DataFrame, min_denom: int = 20, min_players: int = 10,
) -> pl.DataFrame:
    contexts = ["season", "level_id"]
    qualifying = player.filter(pl.col("n") >= min_denom)
    fitted = qualifying.group_by(contexts).agg(
        pl.col("stat").var().alias("var_obs"), pl.len().alias("n_players"),
        (1 / pl.col("n")).mean().alias("mean_inv_n"),
    )
    within = qualifying.filter(pl.col("n") > 1).group_by(contexts).agg(
        ((pl.col("var") * (pl.col("n") - 1)).sum()
         / (pl.col("n") - 1).sum()).alias("var_within")
    )
    fitted = fitted.join(within, on=contexts, how="left").with_columns(
        (pl.col("var_obs") - pl.col("var_within") * pl.col("mean_inv_n"))
        .alias("var_between_raw")
    ).with_columns(
        pl.when(pl.col("var_between_raw").is_not_null())
        .then(pl.max_horizontal(pl.lit(1e-8), pl.col("var_between_raw")))
        .otherwise(None).alias("var_between")
    ).with_columns(
        pl.when(pl.col("var_between").is_not_null())
        .then(pl.max_horizontal(pl.lit(0.0), pl.col("var_within") / pl.col("var_between")))
        .otherwise(None).alias("K")
    ).with_columns(
        pl.col("K").alias("n_50"), (pl.col("K") * 7 / 3).alias("n_70"),
        (pl.col("K") * 4).alias("n_80"),
    ).filter(pl.col("n_players") >= min_players)
    # All recorded swings contribute to league COE, including small player samples.
    league = player.group_by(contexts).agg(
        ((pl.col("stat") * pl.col("n")).sum() / pl.col("n").sum()).alias("mu")
    )
    return fitted.join(league, on=contexts, validate="1:1").with_columns(
        pl.lit("hitters").alias("dataset"), pl.lit("contact_vs_avg").alias("stat"),
        pl.lit("mean").alias("kind"), pl.lit(None, dtype=pl.String).alias("pitch_tag"),
    )


def align_constants(rows: pl.DataFrame, existing: pl.DataFrame) -> pl.DataFrame:
    return rows.select([
        (pl.col(c) if c in rows.columns else pl.lit(None)).cast(dtype, strict=False).alias(c)
        for c, dtype in existing.schema.items()
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-path", type=Path, nargs="+", required=True)
    parser.add_argument("--constants", type=Path, default=REPO_DIR / "config/stability_constants.csv")
    args = parser.parse_args()
    players = pl.concat([contact_players(pl.scan_parquet(p)) for p in args.parquet_path], how="vertical_relaxed")
    if players.select(["batter_mlbid", "season", "level_id"]).is_duplicated().any():
        raise ValueError("Overlapping hitter-season-level sources; provide disjoint season files.")
    rows = contact_constants(players)
    existing = pl.read_csv(args.constants)
    # Carry prior-season K into the next season, matching the daily blend convention.
    prior = rows.select(
        (pl.col("season") + 1).alias("season"), "level_id", pl.col("K").alias("prior_K")
    )
    target_season = rows["season"].max()
    rows = rows.join(prior, on=["season", "level_id"], how="left").with_columns(
        pl.when(pl.col("season") == target_season)
        .then(pl.coalesce("prior_K", "K")).otherwise(pl.col("K")).alias("K")
    ).with_columns(pl.col("K").alias("n_50"), (pl.col("K") * 7 / 3).alias("n_70"),
                   (pl.col("K") * 4).alias("n_80"))
    kept = existing.filter(~(
        (pl.col("dataset") == "hitters") & (pl.col("stat") == "contact_vs_avg")
        & pl.col("season").is_in(players["season"].unique().to_list())
    ).fill_null(False))
    updated = pl.concat([kept, align_constants(rows, existing)])
    updated.write_csv(args.constants)
    updated.write_parquet(args.constants.with_suffix(".parquet"))
    print(f"Wrote {rows.height} direct COE priors; other constants preserved.")


if __name__ == "__main__":
    main()
