"""Build compact regular-season hitter pitch-group popout data from saved pitches."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hitter_pitch_splits import COLUMNS
from pipeline.pitch_split_games import verified_games, apply_game_classifications
from pipeline.data_aggregate import (
    _add_game_type_group, _normalize_age_columns, _normalize_player_names, _normalize_team_codes, build_hitters,
)

KEYS = ["batter_mlbid", "season", "level_id"]
SCOPE_KEYS = {"hitter": KEYS, "team": ["hitting_code", "season", "level_id"],
              "league": ["season", "level_id"]}
SOURCE_COLUMNS = """batter_mlbid hitter_name level_id season game_type batter_position
batter_hand pa_id bbe exit_velo launch_angle spray_angle_adj is_in_play swing whiff
is_inzone_pi pi_pitch_group pi_pitch_type pitch_velo damage_pred decision_value pred_whiff_loc
miss_distance bat_speed swing_length attack_angle attack_direction swing_path_tilt
intercept_ball_minus_batter_pos_x_inches intercept_ball_minus_batter_pos_y_inches
hitting_code pitch_outcome game_pk batter_age""".split()
OPTIONAL_TRACKING = """miss_distance bat_speed swing_length attack_angle attack_direction swing_path_tilt
intercept_ball_minus_batter_pos_x_inches intercept_ball_minus_batter_pos_y_inches""".split()


def build_splits(pitch, scope="hitter"):
    keys = SCOPE_KEYS[scope]
    if "game_type" not in pitch.columns:
        raise ValueError("Pitch type splits require explicit game_type values.")
    pitch = pitch.filter(pl.col("game_type") == "R")
    pitch = _normalize_age_columns(_normalize_player_names(_add_game_type_group(pitch)))
    pitch = pitch.filter(pl.col("game_type_group") == "Regular Season")
    if pitch.is_empty():
        return pl.DataFrame()
    if scope == "team":
        pitch = _normalize_team_codes(pitch, "hitting_code").with_columns(
            pl.col("hitting_code").replace("OAK", "ATH")
        ).filter(pl.col("hitting_code").is_not_null() & ~pl.col("hitting_code").str.contains(r"^\d+$"))
        if pitch.is_empty():
            return pl.DataFrame()
    # Source name variants must not split a single player's pitch-group totals.
    pitch = pitch.with_columns(pl.col("hitter_name").first().over(KEYS).alias("hitter_name"))
    pitch = pitch.with_columns([
        (pl.col(c).cast(pl.Float64, strict=False) if c in pitch.columns else pl.lit(None, dtype=pl.Float64)).alias(c)
        for c in OPTIONAL_TRACKING
    ])
    totals = pitch.group_by(keys).agg(pl.len().alias("total_pitches"))
    # Use source type, so SI pitches tagged SP still belong with sinkers.
    # Only cutters classified as fastballs belong in the Fastball row.
    pitch = pitch.with_columns(
        pl.when((pl.col("pi_pitch_group") == "FA") & pl.col("pi_pitch_type").is_in(["SI", "FT"]))
        .then(pl.lit("SI"))
        .when((pl.col("pi_pitch_group") == "FA") & pl.col("pi_pitch_type").is_in(["FA", "FC"]))
        .then(pl.lit("FA"))
        .when(pl.col("pi_pitch_group").is_in(["SL", "CU"]))
        .then(pl.lit("BR"))
        .when(pl.col("pi_pitch_group") == "CH")
        .then(pl.lit("OFF"))
        .otherwise(None).alias("__popout_group")
    )
    frames = []
    for group in ["FA", "SI", "BR", "OFF"]:
        subset = pitch.filter(pl.col("__popout_group") == group)
        if subset.is_empty():
            continue
        frame = build_hitters(subset, group_keys=keys if scope != "hitter" else None).join(totals, on=keys)
        frame = frame.with_columns(
            pl.lit(group).alias("pitch_group"),
            (100 * pl.col("pitches") / pl.col("total_pitches")).alias("pitch_pct"),
            pl.when(pl.col("whiff_rate_den") > 0)
            .then(100 * pl.col("whiff_rate_num") / pl.col("whiff_rate_den"))
            .otherwise(None).alias("whiff_pct"),
        )
        frames.append(frame.select(keys + ["pitch_group", "PA"] + list(COLUMNS) + ["miss_distance_n"]))
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--seasons", type=int, nargs="+")
    parser.add_argument("--scope", choices=list(SCOPE_KEYS), default="hitter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    keys = SCOPE_KEYS[args.scope]
    if args.output is None:
        name = "hitter_pitch_splits" if args.scope == "hitter" else f"{args.scope}_hitting_pitch_splits"
        args.output = ROOT / "data/output" / f"{name}.parquet"
    frames, replaced = [], []
    scans = [pl.scan_parquet(path) for path in args.input]
    verified = verified_games(scans)
    for scan in scans:
        scan = apply_game_classifications(scan, verified)
        schema = scan.collect_schema()
        seasons = scan.select("season").unique().collect()["season"].to_list()
        for season in sorted(seasons):
            if args.seasons and season not in args.seasons:
                continue
            # Load one level at a time to keep the historical backfill bounded.
            season_scan = scan.filter(pl.col("season") == season)
            levels = season_scan.select("level_id").unique().collect()["level_id"].to_list()
            for level in levels:
                pitch = season_scan.filter(pl.col("level_id") == level).select(
                    [c for c in SOURCE_COLUMNS if c in schema]
                ).collect()
                result = build_splits(pitch, args.scope)
                replaced.append((season, level))
                if not result.is_empty():
                    frames.append(result)
                print(f"Built {season}, level {level}: {len(result):,} rows", flush=True)
    if not frames:
        raise ValueError("No pitch-group rows built; existing output was preserved.")
    if args.output.exists():
        old = pl.read_parquet(args.output)
        replaced_keys = pl.DataFrame(replaced, schema=["season", "level_id"], orient="row").cast(
            {c: old.schema[c] for c in ["season", "level_id"]}
        )
        frames.insert(0, old.join(replaced_keys, on=["season", "level_id"], how="anti"))
    result = pl.concat(frames, how="diagonal_relaxed").sort(keys + ["pitch_group"])
    if result.select(keys + ["pitch_group"]).is_duplicated().any():
        raise ValueError("Duplicate hitter pitch-group identities; existing output was preserved.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        import shutil
        backup = ROOT / "data/backups/pitch_split_level_correction" / args.output.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(args.output, backup)
    temp = args.output.with_suffix(".tmp.parquet")
    result.write_parquet(temp)
    temp.replace(args.output)
    print(f"Wrote {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
