"""Fit missing Comps rate priors from retained season-level counts."""
from pathlib import Path

import polars as pl
import yaml

from model_dev.stability_constants import rate_constants


def comparison_rate_counts(df, dataset):
    """Alias retained counts for the standard apply_rate_from_agg path; no fitting."""
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/comparison_rates.yml").read_text())
    aliases = {"rate_whiffs": ["whiff_rate_num", "SwStr_num"],
               "rate_swings": ["whiff_rate_den", "Swing_pct_num"],
               "rate_pitches": ["Swing_pct_den", "SwStr_den", "pitches"]}
    expressions = []
    for stat, columns in config["stats"].get(dataset, {}).items():
        for suffix, column in zip(["num", "den"], columns):
            candidates = [c for c in [column, *aliases.get(column, [])] if c in df.columns]
            if not candidates:
                raise ValueError(f"Missing Comps regression counts for {dataset}.{stat}")
            expressions.append(pl.coalesce([pl.col(c).cast(pl.Float64) for c in candidates]).alias(f"{stat}_{suffix}"))
    return df.with_columns(expressions)


def historical_rate_constants(df, dataset):
    """One-time initialization using the same estimator as existing rate stats."""
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/comparison_rates.yml").read_text())
    df = comparison_rate_counts(df, dataset).filter(pl.col("game_type_group") == "Regular Season")
    contexts = ["season", "level_id"]
    identity = "batter_mlbid" if dataset == "hitters" else "pitcher_mlbid"
    rows = []
    for stat in config["stats"].get(dataset, {}):
        player = df.group_by([identity, *contexts]).agg(
            pl.col(stat + "_num").sum().alias("num"), pl.col(stat + "_den").sum().alias("den"))
        player = player.filter((pl.col("den") >= config["min_opportunities"])
            & pl.col("num").is_between(0, pl.col("den"))).with_columns(
                (pl.col("num") / pl.col("den")).alias("stat"))
        rows.append(rate_constants(player, contexts, config["min_players"]).with_columns(
            pl.lit(dataset).alias("dataset"), pl.lit(stat).alias("stat"),
            pl.lit("rate").alias("kind"), pl.lit(None, dtype=pl.String).alias("pitch_tag")))
    return pl.concat(rows, how="diagonal_relaxed")
