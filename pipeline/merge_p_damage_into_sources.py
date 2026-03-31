from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


PITCHER_SOURCE = "pitcher_p_damage.parquet"
PITCH_TYPE_SOURCE = "pitch_types_p_damage.parquet"


def _weighted_pct_merge(
    df: pl.DataFrame,
    *,
    keys: list[str],
    weight_col: str,
    value_cols: list[str],
) -> pl.DataFrame:
    agg_exprs: list[pl.Expr] = [pl.col(weight_col).sum().alias(weight_col)]
    for col in value_cols:
        agg_exprs.append(
            ((pl.col(col) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()).alias(col)
        )
    return df.group_by(keys).agg(agg_exprs)


def _fill_game_type_group(df: pl.DataFrame) -> pl.DataFrame:
    """Fill NULL game_type_group — historical season chunks pre-date this column."""
    if "game_type_group" in df.columns:
        return df.with_columns(pl.col("game_type_group").fill_null("Regular Season"))
    return df


def _merge_pitchers(pitchers: pl.DataFrame, source_dir: Path) -> pl.DataFrame:
    src_path = source_dir / PITCHER_SOURCE
    if not src_path.exists():
        return pitchers
    pitchers = _fill_game_type_group(pitchers)
    src = pl.read_parquet(src_path).rename({"pitcher_name": "name"})
    join_keys = ["pitcher_mlbid", "season"]
    if "level_id" in src.columns:
        join_keys.append("level_id")
    if "game_type_group" in src.columns:
        join_keys.append("game_type_group")
    src = _weighted_pct_merge(
        src,
        keys=join_keys,
        weight_col="n",
        value_cols=["Damage_pct", "p_Damage_pct", "p_Damage_with_loc_pct"],
    ).rename({"n": "p_Damage_n"}).with_columns(pl.col("p_Damage_n").alias("Damage_n"))
    return pitchers.join(src, on=join_keys, how="left")


def _merge_pitch_types(pitch_types: pl.DataFrame, source_dir: Path) -> pl.DataFrame:
    src_path = source_dir / PITCH_TYPE_SOURCE
    if not src_path.exists():
        return pitch_types
    pitch_types = _fill_game_type_group(pitch_types)
    src = pl.read_parquet(src_path).rename({"pitcher_name": "name"})
    join_keys = ["pitcher_mlbid", "season", "pitch_tag"]
    if "level_id" in src.columns:
        join_keys.append("level_id")
    if "game_type_group" in src.columns:
        join_keys.append("game_type_group")
    src = _weighted_pct_merge(
        src,
        keys=join_keys,
        weight_col="n",
        value_cols=["Damage_pct", "p_Damage_pct", "p_Damage_with_loc_pct"],
    ).rename({"n": "p_Damage_n"}).with_columns(pl.col("p_Damage_n").alias("Damage_n"))
    return pitch_types.join(src, on=join_keys, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge p(damage) source tables into aggregate source files.")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw")
    parser.add_argument("--pitchers", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "output" / "pitcher_stuff_new.parquet")
    parser.add_argument("--pitch-types", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "output" / "new_pitch_types.parquet")
    args = parser.parse_args()

    pitchers = pl.read_parquet(args.pitchers)
    pitch_types = pl.read_parquet(args.pitch_types)
    drop_cols = ["Damage_pct", "Damage_n", "p_Damage_n", "p_Damage_pct", "p_Damage_with_loc_pct"]
    pitchers = pitchers.drop([c for c in drop_cols if c in pitchers.columns])
    pitch_types = pitch_types.drop([c for c in drop_cols if c in pitch_types.columns])
    pitchers = _merge_pitchers(pitchers, args.source_dir)
    pitch_types = _merge_pitch_types(pitch_types, args.source_dir)
    pitchers.write_parquet(args.pitchers)
    pitch_types.write_parquet(args.pitch_types)
    print(f"Updated {args.pitchers}")
    print(f"Updated {args.pitch_types}")


if __name__ == "__main__":
    main()
