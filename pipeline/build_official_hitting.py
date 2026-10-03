"""Preload official regular-season totals using the local scraper package."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/output/official_hitting.parquet"
FIELDS = {
    "games_played": "GP", "plate_appearances": "PA", "at_bats": "AB",
    "avg": "AVG", "obp": "OBP", "slg": "SLG", "ops": "OPS",
    "strike_outs": "SO", "base_on_balls": "BB", "home_runs": "HR",
    "hits": "H", "runs": "R", "rbi": "RBI", "stolen_bases": "SB",
    "caught_stealing": "CS", "babip": "BABIP",
}


def normalize_totals(frame, level_id=1):
    required = ["player_id", "season", "player_name", *FIELDS]
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Missing official hitting data: {sorted(missing)}")
    out = frame[required].rename(columns=FIELDS).copy()
    if out.duplicated(["player_id", "season"]).any():
        raise ValueError("Expected one all-team total per player-season")
    out["player_name"] = out.player_name.map(
        lambda name: unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    )
    for col in FIELDS.values():
        out[col] = pd.to_numeric(out[col].replace({".---": None, "---": None, "-.--": None}), errors="raise")
    out["level_id"] = level_id
    # Standard and advanced StatsAPI responses do not publish an HR/FB rate.
    out["HR/FB%"] = float("nan")
    out["K%"] = 100 * out.SO / out.PA.where(out.PA > 0)
    out["BB%"] = 100 * out.BB / out.PA.where(out.PA > 0)
    out["retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return out


def fetch_season(client, season, level_id=1):
    frame = client.season_stats(season, sport_id=level_id, refresh=True, page_size=1000).tables["season_stats"]
    if frame.empty:
        return pd.DataFrame()
    # A player traded between teams can have multiple source splits. Retrieve
    # the authoritative combined total rather than adding overlapping rows.
    duplicate_ids = frame.loc[frame.duplicated("player_id", keep=False), "player_id"].unique()
    parts = [frame[~frame.player_id.isin(duplicate_ids)]]
    for player_id in duplicate_ids:
        splits = client.season_stats(season, sport_id=level_id, player_id=int(player_id), refresh=True).tables["season_stats"]
        if len(splits) > 1:
            splits = splits[pd.to_numeric(splits.num_teams, errors="coerce") > 1]
        if len(splits) != 1:
            raise ValueError(f"Ambiguous combined total: {season}, player {player_id}")
        parts.append(splits)
    sys.path.insert(0, str(ROOT))
    from pipeline.official_sabermetrics import fetch_sabermetric
    totals = normalize_totals(pd.concat(parts, ignore_index=True), level_id)
    metric = fetch_sabermetric(client, season, level_id, "hitting")
    return totals.merge(metric, on="player_id", how="left", validate="one_to_one")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 11, 14, 16])
    parser.add_argument("--scraper-path", type=Path, default=ROOT.parent / "codex_tmp/scraper")
    args = parser.parse_args()
    sys.path.insert(0, str(args.scraper_path))
    from baseball_data import BaseballClient

    client = BaseballClient()
    frames = []
    for season in args.seasons:
        for level_id in args.levels:
            frame = fetch_season(client, season, level_id)
            if not frame.empty:
                frames.append(frame)
            print(f"Official hitting {season}, level {level_id}: {len(frame)} players", flush=True)
    if OUTPUT.exists():
        previous = pd.read_parquet(OUTPUT)
        if "level_id" not in previous:
            previous["level_id"] = 1
        frames.append(previous[~(previous.season.isin(args.seasons) & previous.level_id.isin(args.levels))])
    result = pd.concat(frames, ignore_index=True)
    assert not result.duplicated(["player_id", "season", "level_id"]).any()
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT)


if __name__ == "__main__":
    main()
