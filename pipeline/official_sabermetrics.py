"""Shared player sabermetric API pull for daily refresh."""
import pandas as pd


def fetch_sabermetric(client, season, level_id, group):
    """Fetch one official all-team player metric with complete pagination."""
    field, column = ("wRcPlus", "wRC+") if group == "hitting" else ("xfip", "xFIP")
    rows, offset, expected_total = [], 0, None
    for _ in range(1000):
        url = (
            "https://statsapi.mlb.com/api/v1/stats?stats=sabermetrics"
            f"&group={group}&season={season}&sportIds={level_id}"
            f"&gameType=R&playerPool=ALL&limit=1000&offset={offset}"
        )
        payload, _ = client._get(url, True)
        blocks = payload.get("stats", [])
        if not blocks and offset == 0:
            return pd.DataFrame(columns=["player_id", column])
        if len(blocks) != 1 or blocks[0].get("group", {}).get("displayName") != group:
            raise ValueError("Unexpected sabermetric response")
        block = blocks[0]
        total = block.get("totalSplits")
        if not isinstance(total, int) or (expected_total is not None and total != expected_total):
            raise ValueError("Sabermetric pagination total missing or changed")
        expected_total = total
        splits = block.get("splits", [])
        for split in splits:
            if (str(split.get("season")) != str(season)
                    or split.get("sport", {}).get("id", level_id) != level_id
                    or split.get("gameType", "R") != "R"):
                raise ValueError("Sabermetric identity mismatch")
            stat = split["stat"]
            rows.append({"player_id": split["player"]["id"],
                         column: stat.get(field), "num_teams": split.get("numTeams", 1)})
        offset += len(splits)
        if offset == total:
            break
        if not splits or offset > total:
            raise ValueError("Incomplete sabermetric pagination")
    else:
        raise ValueError("Sabermetric page limit exceeded")
    frame = pd.DataFrame(rows, columns=["player_id", column, "num_teams"])
    player_ids = set(frame.player_id)
    duplicates = frame.duplicated("player_id", keep=False)
    frame = frame[~duplicates | (pd.to_numeric(frame.num_teams, errors="coerce") > 1)]
    if frame.duplicated("player_id").any() or set(frame.player_id) != player_ids:
        raise ValueError("Ambiguous sabermetric season totals")
    frame[column] = pd.to_numeric(frame[column].replace({"-": None, ".---": None}), errors="raise")
    return frame.drop(columns="num_teams")
