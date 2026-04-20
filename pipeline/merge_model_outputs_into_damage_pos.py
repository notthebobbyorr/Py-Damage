"""
Merge batter-level p(swing), p(swstr), and p(damage) model outputs
into the damage_pos aggregated hitter files.

Reads:
  data/raw/hitter_p_swstr.parquet   – batter swing/swstr columns
  data/raw/hitter_p_damage.parquet  – batter damage columns
  data/output/damage_pos_2015_YYYY.parquet  – latest stitched hitter file

Writes:
  data/output/damage_pos_2015_YYYY.parquet  – same file, with new model columns

Columns added (or refreshed) to damage_pos:
  SwStr_pct, Swing_pct
  p_SwStr_pct, p_SwStr_with_loc_pct, p_SwStr_n
  p_Swing_pct, p_Swing_with_loc_pct, p_Swing_n   (p_Swing_n aliases p_SwStr_n)
  Damage_pct
  p_Damage_pct, p_Damage_with_loc_pct, p_Damage_n

Join key: batter_mlbid + season + level_id + game_type_group
"""
from __future__ import annotations

from pathlib import Path
import polars as pl

HERE = Path(__file__).resolve().parent.parent
RAW_DIR = HERE / "data" / "raw"
OUTPUT_DIR = HERE / "data" / "output"

SWSTR_SOURCE = "hitter_p_swstr.parquet"
DAMAGE_SOURCE = "hitter_p_damage.parquet"

SWSTR_COLS = ["SwStr_pct", "Swing_pct", "p_SwStr_pct", "p_SwStr_with_loc_pct", "p_Swing_pct", "p_Swing_with_loc_pct"]
DAMAGE_COLS = ["Damage_pct", "p_Damage_pct", "p_Damage_with_loc_pct"]


def _find_latest_damage_pos(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("damage_pos_2015_[0-9][0-9][0-9][0-9].parquet"))
    return candidates[-1] if candidates else None


def _fill_game_type_group(df: pl.DataFrame) -> pl.DataFrame:
    """Fill NULL game_type_group — historical season chunks pre-date this column."""
    if "game_type_group" in df.columns:
        return df.with_columns(pl.col("game_type_group").fill_null("Regular Season"))
    return df


def _weighted_pct_merge(
    src: pl.DataFrame,
    keys: list[str],
    weight_col: str,
    value_cols: list[str],
) -> pl.DataFrame:
    """Weighted-average value_cols by weight_col, grouped by keys."""
    available = [c for c in value_cols if c in src.columns]
    if not available:
        return src.select(keys + [pl.col(weight_col)])
    weighted = [pl.col(c) * pl.col(weight_col) for c in available]
    agg = (
        src.with_columns(weighted)
        .group_by(keys)
        .agg(
            [pl.col(c).sum() for c in available] + [pl.col(weight_col).sum()]
        )
    )
    for c in available:
        agg = agg.with_columns((pl.col(c) / pl.col(weight_col)).alias(c))
    return agg


def _merge_swstr(damage_pos: pl.DataFrame, source_dir: Path) -> pl.DataFrame:
    src_path = source_dir / SWSTR_SOURCE
    if not src_path.exists():
        print(f"  Warning: {src_path} not found — skipping swstr/swing merge.")
        return damage_pos

    damage_pos = _fill_game_type_group(damage_pos)
    src = pl.read_parquet(src_path)
    join_keys = ["batter_mlbid", "season"]
    for col in ["level_id", "game_type_group"]:
        if col in src.columns:
            join_keys.append(col)

    src = _weighted_pct_merge(
        src,
        keys=join_keys,
        weight_col="n",
        value_cols=SWSTR_COLS,
    ).rename({"n": "p_SwStr_n"}).with_columns([
        pl.col("p_SwStr_n").alias("p_Swing_n"),
        pl.col("p_SwStr_n").alias("Swing_n"),
    ])
    return damage_pos.join(src, on=join_keys, how="left")


def _merge_damage(damage_pos: pl.DataFrame, source_dir: Path) -> pl.DataFrame:
    src_path = source_dir / DAMAGE_SOURCE
    if not src_path.exists():
        print(f"  Warning: {src_path} not found — skipping damage merge.")
        return damage_pos

    damage_pos = _fill_game_type_group(damage_pos)
    src = pl.read_parquet(src_path)
    join_keys = ["batter_mlbid", "season"]
    for col in ["level_id", "game_type_group"]:
        if col in src.columns:
            join_keys.append(col)

    src = _weighted_pct_merge(
        src,
        keys=join_keys,
        weight_col="n",
        value_cols=DAMAGE_COLS,
    ).rename({"n": "p_Damage_n"})
    return damage_pos.join(src, on=join_keys, how="left")


def main() -> None:
    path = _find_latest_damage_pos(OUTPUT_DIR)
    if path is None:
        raise FileNotFoundError(f"No damage_pos_2015_YYYY.parquet found in {OUTPUT_DIR}")

    print(f"Loading {path.name} ...")
    damage_pos = pl.read_parquet(path)
    print(f"  Rows: {len(damage_pos):,}  Cols: {len(damage_pos.columns)}")

    # Drop any previously merged model columns so re-runs are clean
    drop_cols = SWSTR_COLS + ["p_SwStr_n", "p_Swing_n", "Swing_n"] + DAMAGE_COLS + ["p_Damage_n"]
    damage_pos = damage_pos.drop([c for c in drop_cols if c in damage_pos.columns])

    damage_pos = _merge_swstr(damage_pos, RAW_DIR)
    damage_pos = _merge_damage(damage_pos, RAW_DIR)

    print(f"  Cols after merge: {len(damage_pos.columns)}")
    damage_pos.write_parquet(path)
    print(f"Wrote {len(damage_pos):,} rows to {path.name}")


if __name__ == "__main__":
    main()
