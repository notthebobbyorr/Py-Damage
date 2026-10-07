"""Correct saved position counts without changing PA-based eligibility flags."""
from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd
import polars as pl

KEYS = ["batter_mlbid", "level_id", "season", "game_type_group"]
POSITIONS = ["UT", "C", "X1B", "X2B", "X3B", "SS", "OF", "P", "NA"]


def count_position_games(raw):
    mapping = {1: "P", 2: "C", 3: "X1B", 4: "X2B", 5: "X3B", 6: "SS",
               7: "OF", 8: "OF", 9: "OF", 10: "UT", 11: "UT", 12: "UT"}
    raw = raw.with_columns(
        pl.col("batter_position").replace_strict(mapping, default="NA").alias("position"),
        pl.when(pl.col("game_type") == "S").then(pl.lit("Spring Training"))
        .when(pl.col("game_type").is_in(["F", "D", "L", "W"]))
        .then(pl.lit("Postseason")).otherwise(pl.lit("Regular Season")).alias("game_type_group"),
    )
    games = (raw.filter(pl.col("game_pk").is_not_null()).group_by(KEYS + ["position"])
             .agg(pl.col("game_pk").n_unique().alias("games"))
             .pivot(values="games", index=KEYS, on="position").fill_null(0))
    for col in POSITIONS:
        if col not in games.columns:
            games = games.with_columns(pl.lit(0).alias(col))
    return games.select(KEYS + POSITIONS).to_pandas()


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/raw"
    inputs = sorted((raw_dir / "_hist_seasons").glob("pitch_data_*.parquet"))
    inputs += [raw_dir / "pitch_data_2026.parquet"]
    frames = []
    for path in inputs:
        raw = pl.read_parquet(path, columns=["batter_mlbid", "level_id", "season", "game_type", "game_pk", "batter_position"])
        frames.append(count_position_games(raw))
        print(f"Counted {path.name}", flush=True)
    counts = pd.concat(frames, ignore_index=True)
    assert not counts.duplicated(KEYS).any()
    output = root / "data/output"
    targets = sorted(output.glob("damage_pos_*.parquet")) + [output / "hitter_pctiles.parquet"]
    targets += sorted((output / "_season_chunks").glob("damage_pos_*.parquet"))
    targets += sorted((output / "_season_chunks").glob("hitter_pctiles*.parquet"))
    backup = root / "data/backups" / ("position_games_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    for path in targets:
        data = pd.read_parquet(path)
        keys = data[KEYS].copy()
        keys["game_type_group"] = keys["game_type_group"].astype("string").fillna("Regular Season")
        matched = keys.merge(counts, on=KEYS, how="left", validate="many_to_one", indicator=True)
        if (matched["_merge"] != "both").any():
            raise ValueError(f"Unmatched rows in {path}; refusing to invent zero counts")
        for col in POSITIONS:
            data[col] = matched[col].to_numpy(dtype="int32")
        saved = backup / path.relative_to(output)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        data.to_parquet(path, index=False)
        print(f"Updated {path.relative_to(output)} ({len(data)} rows)", flush=True)
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
