"""Preload official regular-season totals using the local scraper package."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/output/official_pitching.parquet"
FIELDS = {
    "games_pitched": "G", "games_started": "GS", "innings_pitched": "IP",
    "wins": "W", "losses": "L", "saves": "SV", "era": "ERA", "whip": "WHIP",
    "avg": "AVG", "batters_faced": "TBF", "strike_outs": "SO",
    "base_on_balls": "BB", "home_runs": "HR", "hits": "HA",
    "ops": "OPS", "babip": "BABIP", "quality_starts": "QS",
    "blown_saves": "BS", "holds": "H",
    "inherited_runners": "IR", "inherited_runners_scored": "IRS",
}


def normalize_totals(frame, level_id=1):
    required = ["player_id", "season", "player_name", *FIELDS]
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Missing official pitching data: {sorted(missing)}")
    out = frame[required].rename(columns=FIELDS).copy()
    if out.duplicated(["player_id", "season"]).any():
        raise ValueError("Expected one all-team total per player-season")
    out["player_name"] = out.player_name.map(
        lambda name: unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    )
    out["IP"] = out["IP"].astype(str)
    if not out["IP"].str.fullmatch(r"\d+\.[012]").all():
        raise ValueError("Invalid baseball innings notation")
    for col in [c for c in FIELDS.values() if c != "IP"]:
        out[col] = pd.to_numeric(out[col].replace({"-": None, ".---": None, "---": None, "-.--": None}), errors="raise")
    out["level_id"] = level_id
    out["K%"] = 100 * out.SO / out.TBF.where(out.TBF > 0)
    out["BB%"] = 100 * out.BB / out.TBF.where(out.TBF > 0)
    out["K-BB%"] = out["K%"] - out["BB%"]
    out["Inh. Runners %"] = 100 * out.IRS / out.IR.where(out.IR > 0)
    out["retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return out


def fetch_advanced(client, season, level_id):
    """Read official QS/BABIP with complete pagination; never infer missing QS."""
    rows, offset, expected_total = [], 0, None
    for _ in range(1000):
        url = (
            "https://statsapi.mlb.com/api/v1/stats?stats=seasonAdvanced"
            f"&group=pitching&season={season}&sportIds={level_id}"
            f"&gameType=R&playerPool=ALL&limit=1000&offset={offset}"
        )
        payload, _ = client._get(url, True)
        blocks = payload.get("stats", [])
        if not blocks and offset == 0:
            return pd.DataFrame(columns=["player_id", "quality_starts", "babip"])
        if len(blocks) != 1 or blocks[0].get("group", {}).get("displayName") != "pitching":
            raise ValueError("Unexpected advanced pitching response")
        block = blocks[0]
        total = block.get("totalSplits")
        if not isinstance(total, int) or (expected_total is not None and total != expected_total):
            raise ValueError("Advanced pitching pagination total missing or changed")
        expected_total = total
        splits = block.get("splits", [])
        for split in splits:
            if (str(split.get("season")) != str(season)
                    or split.get("sport", {}).get("id", level_id) != level_id
                    or split.get("gameType", "R") != "R"):
                raise ValueError("Advanced pitching identity mismatch")
            stat = split["stat"]
            rows.append({"player_id": split["player"]["id"],
                         "quality_starts": stat.get("qualityStarts"),
                         "babip": stat.get("babip"), "num_teams": split.get("numTeams", 1)})
        offset += len(splits)
        if offset == total:
            break
        if not splits or offset > total:
            raise ValueError("Incomplete advanced pitching pagination")
    else:
        raise ValueError("Advanced pitching page limit exceeded")
    frame = pd.DataFrame(rows, columns=["player_id", "quality_starts", "babip", "num_teams"])
    player_ids = set(frame.player_id)
    duplicates = frame.duplicated("player_id", keep=False)
    frame = frame[~duplicates | (pd.to_numeric(frame.num_teams, errors="coerce") > 1)]
    if frame.duplicated("player_id").any() or set(frame.player_id) != player_ids:
        raise ValueError("Ambiguous advanced pitching season totals")
    return frame.drop(columns="num_teams")


def fetch_season(client, season, level_id=1):
    frame = client.season_stats(season, group="pitching", sport_id=level_id, refresh=True, page_size=1000).tables["season_stats"]
    if frame.empty:
        return pd.DataFrame()
    # A player traded between teams can have multiple source splits. Retrieve
    # the authoritative combined total rather than adding overlapping rows.
    duplicate_ids = frame.loc[frame.duplicated("player_id", keep=False), "player_id"].unique()
    parts = [frame[~frame.player_id.isin(duplicate_ids)]]
    for player_id in duplicate_ids:
        splits = client.season_stats(season, group="pitching", sport_id=level_id, player_id=int(player_id), refresh=True).tables["season_stats"]
        if len(splits) > 1:
            splits = splits[pd.to_numeric(splits.num_teams, errors="coerce") > 1]
        if len(splits) != 1:
            raise ValueError(f"Ambiguous combined total: {season}, player {player_id}")
        parts.append(splits)
    combined = pd.concat(parts, ignore_index=True)
    advanced = fetch_advanced(client, season, level_id)
    combined = combined.drop(columns=["quality_starts", "babip"], errors="ignore")
    combined = combined.merge(advanced, on="player_id", how="left", validate="one_to_one")
    sys.path.insert(0, str(ROOT))
    from pipeline.official_sabermetrics import fetch_sabermetric
    totals = normalize_totals(combined, level_id)
    metric = fetch_sabermetric(client, season, level_id, "pitching")
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
            print(f"Official pitching {season}, level {level_id}: {len(frame)} players", flush=True)
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
