from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl
import psycopg2

from data_pull import load_db_config


DEFAULT_FILES = [
    "pitch_data_2015_2025_level1.parquet",
    "pitch_data_2021_2025_no_level1.parquet",
]

JOIN_KEYS = ["game_pk", "at_bat_index", "event_index", "pitch_of_ab"]


def _fetch_age_rows(
    min_season: int,
    max_season: int,
    level_ids: Iterable[int],
) -> pl.DataFrame:
    level_clause = ",".join(str(int(x)) for x in level_ids)
    query = f"""
        SELECT
            a.game_pk,
            a.at_bat_index,
            a.event_index,
            a.pitch_of_ab,
            a.season,
            a.level_id,
            coalesce(sc.age_pit_legacy, sc.age_pit) as pitcher_age,
            coalesce(sc.age_bat_legacy, sc.age_bat) as batter_age
        FROM pitchinfo.pitches_public a
        LEFT JOIN savant.savant_pbp sc
            ON a.game_pk = sc.game_pk
            AND a.pitch_of_ab = sc.pitch_number
            AND (a.at_bat_index + 1) = sc.at_bat_number
        WHERE a.season >= {min_season} AND a.season <= {max_season}
            AND a.level_id IN ({level_clause})
            AND a.game_type = 'R'
    """
    cfg = load_db_config()
    with psycopg2.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
    ) as conn:
        df = pd.read_sql_query(query, conn)
    return pl.from_pandas(df)


def _coalesce_age_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "pitcher_age_src" in df.columns:
        if "pitcher_age" in df.columns:
            df = df.with_columns(
                pl.coalesce(["pitcher_age", "pitcher_age_src"]).alias("pitcher_age")
            )
        else:
            df = df.rename({"pitcher_age_src": "pitcher_age"})
    if "batter_age_src" in df.columns:
        if "batter_age" in df.columns:
            df = df.with_columns(
                pl.coalesce(["batter_age", "batter_age_src"]).alias("batter_age")
            )
        else:
            df = df.rename({"batter_age_src": "batter_age"})
    drop_cols = [c for c in ["pitcher_age_src", "batter_age_src"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
    return df


def _prep_age_join_rows(ages: pl.DataFrame) -> pl.DataFrame:
    required_cols = [*JOIN_KEYS, "pitcher_age_src", "batter_age_src"]
    for col in required_cols:
        if col not in ages.columns:
            ages = ages.with_columns(pl.lit(None).alias(col))
    ages = ages.select(required_cols)
    # Keep one row per pitch key to avoid one-to-many joins and duplicate suffix cols.
    ages = ages.group_by(JOIN_KEYS).agg(
        [
            pl.col("pitcher_age_src").drop_nulls().first().alias("pitcher_age_src"),
            pl.col("batter_age_src").drop_nulls().first().alias("batter_age_src"),
        ]
    )
    return ages


def _infer_levels_and_seasons(lf: pl.LazyFrame) -> tuple[list[int], list[int]]:
    levels = (
        lf.select(pl.col("level_id").drop_nulls().unique())
        .collect()
        .get_column("level_id")
        .to_list()
    )
    seasons = (
        lf.select(pl.col("season").drop_nulls().unique())
        .collect()
        .get_column("season")
        .to_list()
    )
    levels = sorted(int(x) for x in levels)
    seasons = sorted(int(x) for x in seasons)
    return levels, seasons


def backfill_file(
    path: Path,
    out_path: Path,
    min_season: int | None,
    max_season: int | None,
    chunk_by_season: bool,
    temp_dir: Path | None,
) -> None:
    lf = pl.scan_parquet(path)
    level_ids, seasons = _infer_levels_and_seasons(lf)
    if not seasons:
        print(f"{path}: no seasons found, skipping")
        return
    season_min = min_season if min_season is not None else min(seasons)
    season_max = max_season if max_season is not None else max(seasons)
    season_range = [s for s in seasons if season_min <= s <= season_max]
    if not season_range:
        print(f"{path}: no seasons in range {season_min}-{season_max}, skipping")
        return

    if not chunk_by_season:
        print(f"Fetching ages for {path.name} ({season_min}-{season_max})...")
        ages = _fetch_age_rows(season_min, season_max, level_ids)
        ages = ages.rename(
            {"pitcher_age": "pitcher_age_src", "batter_age": "batter_age_src"}
        )
        ages = _prep_age_join_rows(ages)
        df = lf.collect()
        df = df.join(ages, on=JOIN_KEYS, how="left")
        df = _coalesce_age_columns(df)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        print(f"Wrote {len(df):,} rows to {out_path}")
        return

    if temp_dir is None:
        temp_dir = out_path.parent / "_age_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for season in season_range:
        print(f"Fetching ages for {path.name} season {season}...")
        ages = _fetch_age_rows(season, season, level_ids)
        ages = ages.rename(
            {"pitcher_age": "pitcher_age_src", "batter_age": "batter_age_src"}
        )
        ages = _prep_age_join_rows(ages)
        df_season = lf.filter(pl.col("season") == season).collect()
        df_season = df_season.join(ages, on=JOIN_KEYS, how="left")
        df_season = _coalesce_age_columns(df_season)
        chunk_path = temp_dir / f"{out_path.stem}.season_{season}.parquet"
        df_season.write_parquet(chunk_path)
        chunk_paths.append(chunk_path)
        print(f"  wrote {len(df_season):,} rows to {chunk_path}")

    print(f"Combining {len(chunk_paths)} chunks into {out_path}...")
    lazy_frames = [pl.scan_parquet(p) for p in chunk_paths]
    combined = pl.concat(lazy_frames, how="diagonal_relaxed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill pitcher_age and batter_age into pitch_data parquet files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"c:\users\orrro\documents\baseball_env"),
        help="Directory containing pitch_data parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: input-dir).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific parquet filenames to process (default: standard pitch_data files).",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Minimum season to include (default: inferred from file).",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=None,
        help="Maximum season to include (default: inferred from file).",
    )
    parser.add_argument(
        "--chunk-by-season",
        action="store_true",
        help="Process one season at a time to reduce memory usage.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Temporary directory for chunked output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input files instead of writing *_with_age.parquet outputs.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir or input_dir
    files = args.files or DEFAULT_FILES

    for name in files:
        path = input_dir / name
        if not path.exists():
            print(f"Missing input file: {path}")
            continue
        if args.overwrite:
            out_path = path
        else:
            out_path = out_dir / f"{path.stem}_with_age{path.suffix}"
        backfill_file(
            path=path,
            out_path=out_path,
            min_season=args.min_season,
            max_season=args.max_season,
            chunk_by_season=args.chunk_by_season,
            temp_dir=args.temp_dir,
        )


if __name__ == "__main__":
    main()
