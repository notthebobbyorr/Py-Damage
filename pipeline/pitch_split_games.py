"""Verify suspect MLB game classifications against public MLB schedule metadata."""
from pathlib import Path

import polars as pl
import requests

CACHE = Path(__file__).resolve().parents[1] / "data/output/pitch_split_game_metadata.parquet"


def verified_games(scans, cache_path=CACHE):
    candidates = pl.concat([
        scan.filter((pl.col("level_id") == 1) &
                    (pl.col("hitting_code").is_null() | pl.col("hitting_code").str.contains(r"^\d+$")))
        .select("game_pk").unique().collect() for scan in scans
    ]).drop_nulls().unique()["game_pk"].to_list()
    schema = {"game_pk": pl.Int64, "verified_level": pl.Int64, "verified_type": pl.String}
    cached = pl.read_parquet(cache_path) if cache_path.exists() else pl.DataFrame(schema=schema)
    original_count = len(cached)
    cached = cached.unique()
    if cached["game_pk"].is_duplicated().any():
        raise ValueError("Conflicting verified game classifications")
    missing = sorted(set(candidates) - set(cached["game_pk"].to_list()))
    rows = []
    with requests.Session() as session:
        for start in range(0, len(missing), 100):
            batch = missing[start:start + 100]
            response = session.get("https://statsapi.mlb.com/api/v1/schedule",
                                   params={"gamePks": ",".join(map(str, batch)), "hydrate": "team"}, timeout=30)
            response.raise_for_status()
            found = set()
            for date in response.json()["dates"]:
                for game in date["games"]:
                    home = game["teams"]["home"]["team"]["sport"]["id"]
                    away = game["teams"]["away"]["team"]["sport"]["id"]
                    if home != away:
                        raise ValueError(f"Ambiguous sport levels for game {game['gamePk']}")
                    rows.append({"game_pk": game["gamePk"], "verified_level": home,
                                 "verified_type": game["gameType"]})
                    found.add(game["gamePk"])
            if found != set(batch):
                raise ValueError(f"Missing metadata for games {sorted(set(batch) - found)}")
            print(f"Verified {min(start + 100, len(missing))}/{len(missing)} suspect games", flush=True)
    if rows:
        cached = pl.concat([cached, pl.DataFrame(rows, schema=schema)]).unique()
    if cached["game_pk"].is_duplicated().any():
        raise ValueError("Conflicting verified game classifications")
    if rows or len(cached) != original_count:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(".tmp.parquet")
        cached.write_parquet(temp)
        temp.replace(cache_path)
    return cached


def apply_game_classifications(scan, verified):
    return scan.join(verified.lazy(), on="game_pk", how="left", validate="m:1").with_columns(
        pl.coalesce("verified_level", "level_id").alias("level_id"),
        pl.coalesce("verified_type", "game_type").alias("game_type"),
    ).drop("verified_level", "verified_type")
