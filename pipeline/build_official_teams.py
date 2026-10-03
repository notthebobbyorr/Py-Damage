"""Preload official team season totals; no API requests at app runtime."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.build_official_hitting import FIELDS as HITTING_FIELDS, normalize_totals as hitting_totals
from pipeline.build_official_pitching import FIELDS as PITCHING_FIELDS, normalize_totals as pitching_totals

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/output/official_teams.parquet"


def fetch_stats(client, season, level_id, group, stat_type):
    rows, offset = [], 0
    expected_total = None
    for _ in range(1000):
        url = ("https://statsapi.mlb.com/api/v1/teams/stats"
               f"?stats={stat_type}&group={group}&season={season}&sportIds={level_id}"
               f"&gameType=R&limit=1000&offset={offset}")
        payload, _ = client._get(url, True)
        blocks = payload.get("stats", [])
        if not blocks and offset == 0:
            return []
        if len(blocks) != 1 or blocks[0]["group"]["displayName"] != group:
            raise ValueError("Unexpected team statistics response")
        block = blocks[0]
        total = block.get("totalSplits")
        if not isinstance(total, int) or (expected_total is not None and total != expected_total):
            raise ValueError("Team pagination total missing or changed")
        expected_total = total
        splits = block.get("splits", [])
        for split in splits:
            if str(split["season"]) != str(season) or split.get("gameType", "R") != "R":
                raise ValueError("Team statistics identity mismatch")
        rows.extend(splits)
        offset += len(splits)
        if offset == total:
            break
        if not splits or offset > total:
            raise ValueError("Incomplete team pagination")
    else:
        raise ValueError("Team pagination limit exceeded")
    ids = [r["team"]["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate team totals")
    return rows


def normalize_team(frame, level_id, group):
    # Reuse the player format's stat schema, but retain stable team identity.
    normalize = hitting_totals if group == "hitting" else pitching_totals
    out = normalize(frame.rename(columns={"team_id": "player_id", "team_name": "player_name"}), level_id)
    out = out.rename(columns={"player_id": "team_id", "player_name": "team_name"})
    if group == "hitting":
        out["R/G"] = out.R / out.GP.where(out.GP > 0)
    else:
        innings = out.IP.str.split(".", regex=False, expand=True).astype(int)
        outs = innings[0] * 3 + innings[1]
        runs = pd.to_numeric(frame["runs"], errors="raise")
        out["RA9"] = 27 * runs / outs.where(outs > 0)
    out["group"] = group
    return out


def fetch_season(client, season, level_id):
    from baseball_data.metadata import snake_case
    teams = client.teams(season, sport_id=level_id, refresh=True).tables["teams"]
    metadata = {int(r.team_id): json.loads(r.source_record_json) for r in teams.itertuples()}
    frames = []
    for group, fields in [("hitting", HITTING_FIELDS), ("pitching", PITCHING_FIELDS)]:
        standard = fetch_stats(client, season, level_id, group, "season")
        if not standard:
            continue
        advanced = {r["team"]["id"]: r["stat"] for r in fetch_stats(client, season, level_id, group, "seasonAdvanced")}
        rows, codes = [], []
        for split in standard:
            team_id = split["team"]["id"]
            meta = metadata[team_id]
            stats = {snake_case(k): v for k, v in {**advanced.get(team_id, {}), **split["stat"]}.items()}
            row = {k: stats.get(k) for k in fields}
            row.update(team_id=team_id, team_name=split["team"]["name"], season=season, runs=stats.get("runs"))
            rows.append(row)
            codes.append(meta.get("abbreviation", str(team_id)))
        out = normalize_team(pd.DataFrame(rows), level_id, group)
        out["team_code"] = codes
        out["team_code"] = out.team_code.replace({"AZ": "ARI"})
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 11, 14, 16])
    parser.add_argument("--scraper-path", type=Path, default=ROOT.parent / "codex_tmp/scraper")
    args = parser.parse_args()
    sys.path.insert(0, str(args.scraper_path))
    from baseball_data import BaseballClient
    client, frames = BaseballClient(), []
    for season in args.seasons:
        for level_id in args.levels:
            frame = fetch_season(client, season, level_id)
            if not frame.empty:
                frames.append(frame)
            print(f"Official teams {season}, level {level_id}: {len(frame)} rows", flush=True)
    if OUTPUT.exists():
        old = pd.read_parquet(OUTPUT)
        frames.append(old[~(old.season.isin(args.seasons) & old.level_id.isin(args.levels))])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["team_id", "season", "level_id", "group"]).any():
        raise ValueError("Duplicate team identities")
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT)


if __name__ == "__main__":
    main()
