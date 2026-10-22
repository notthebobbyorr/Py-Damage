"""Add raw-pitch bat tracking aggregates to saved team/league season tables."""
from pathlib import Path
import argparse
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data_aggregate import _add_game_type_group, _normalize_team_codes, bat_tracking_expressions

SOURCES = ["miss_distance", "bat_speed", "swing_length", "swing_path_tilt", "attack_angle", "attack_direction",
           "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches"]
KEYS = ["season", "level_id", "game_type_group"]


def merge_tracking(existing, tracking, keys):
    if tracking.select(keys).is_duplicated().any():
        raise ValueError("Duplicate tracking identities; existing output preserved.")
    columns = [c for c in tracking.columns if c not in keys]
    types = {k: existing.schema[k] for k in keys}
    return existing.drop([c for c in columns if c in existing.columns]).join(
        tracking.cast(types), on=keys, how="left", validate="m:1", maintain_order="left"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    output = ROOT / "data/output"
    built = {"new_team_damage": [], "new_hitting_lg_avg": []}
    for path in args.input:
        scan = pl.scan_parquet(path)
        schema = scan.collect_schema()
        columns = [c for c in KEYS + ["game_type", "hitting_code"] + SOURCES if c in schema]
        for season in sorted(scan.select("season").unique().collect()["season"].to_list()):
            pitch = _add_game_type_group(scan.filter(pl.col("season") == season).select(columns).collect())
            team_pitch = _normalize_team_codes(pitch, "hitting_code").filter(
                pl.col("hitting_code").is_not_null() & ~pl.col("hitting_code").str.contains(r"^\d+$")
            )
            built["new_team_damage"].append(team_pitch.group_by(["hitting_code"] + KEYS).agg(bat_tracking_expressions(team_pitch)))
            built["new_hitting_lg_avg"].append(pitch.group_by(KEYS).agg(bat_tracking_expressions(pitch)))
            print(f"Aggregated tracking for {season}", flush=True)
    for stem, frames in built.items():
        tracking = pl.concat(frames, how="vertical_relaxed")
        keys = (["hitting_code"] if stem == "new_team_damage" else []) + KEYS
        paths = [output / f"{stem}.parquet"] + sorted((output / "_season_chunks").glob(f"{stem}.season_*.parquet"))
        for path in paths:
            if not path.exists():
                continue
            existing = pl.read_parquet(path)
            result = merge_tracking(existing, tracking, keys)
            # Keep a recoverable copy before replacing existing generated data.
            backup = ROOT / "data/backups/bat_tracking" / path.relative_to(output)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                import shutil
                shutil.copy2(path, backup)
            temp = path.with_suffix(".tracking.tmp.parquet")
            result.write_parquet(temp)
            temp.replace(path)
        print(f"Updated {stem} and season chunks", flush=True)


if __name__ == "__main__":
    main()
