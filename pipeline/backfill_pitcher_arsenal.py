"""Rebuild pitcher arsenal metrics from saved pitch-type aggregates."""
from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd


KEYS = ["pitcher_mlbid", "season", "level_id", "game_type_group", "pitcher_hand", "name"]
METRICS = ["BB_pct", "OFF_pct", "BB_velo", "OFF_velo"]


def build_arsenal(pitches: pd.DataFrame) -> pd.DataFrame:
    totals = pitches.groupby(KEYS, observed=True, dropna=False)["pitches"].sum()
    result = totals.to_frame("total_pitches")
    for label, tags in [("BB", ["SL", "SW", "CU"]), ("OFF", ["CH", "FS"])]:
        group = pitches[pitches["pitch_tag"].isin(tags)]
        counts = group.groupby(KEYS, observed=True, dropna=False)["pitches"].sum()
        result[f"{label}_pct"] = 100 * counts.reindex(result.index, fill_value=0) / totals.where(totals > 0)
        # Ties use pitch tag order; missing velocity stays missing for the primary type.
        primary = group.sort_values(["pitches", "pitch_tag"], ascending=[False, True]).drop_duplicates(KEYS)
        result[f"{label}_velo"] = primary.set_index(KEYS)["velo"].reindex(result.index)
    return result.drop(columns="total_pitches").reset_index()


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "output"
    pitches = pd.read_parquet(data / "new_pitch_types.parquet")
    if pitches.duplicated(KEYS + ["pitch_tag"]).any():
        raise ValueError("Duplicate pitch-type rows: cannot safely calculate arsenal metrics")
    metrics = build_arsenal(pitches)
    target = data / "pitcher_stuff_new.parquet"
    source = pd.read_parquet(target)
    updated = source.drop(columns=METRICS, errors="ignore").merge(metrics, on=KEYS, how="left", validate="many_to_one")
    backup = root / "data" / "backups" / ("pitcher_arsenal_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup.mkdir(parents=True)
    shutil.copy2(target, backup / target.name)
    updated.to_parquet(target, index=False)
    print(f"Updated {len(updated)} pitcher rows; backup: {backup}")
    print(updated.groupby("season", observed=True)[METRICS].count().to_string())


if __name__ == "__main__":
    main()
