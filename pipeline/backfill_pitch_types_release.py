"""
One-time backfill: join rel_z, rel_x, ext, arm_angle (+ _n counters) from
pitcher_stuff_new season chunks onto new_pitch_types season chunks, then
recombine into new_pitch_types.parquet.

Join keys: pitcher_mlbid, season, level_id, game_type_group
"""
from pathlib import Path

import polars as pl

CHUNK_DIR = Path(__file__).parent.parent / "data" / "output" / "_season_chunks"
OUT_DIR = Path(__file__).parent.parent / "data" / "output"

JOIN_KEYS = ["pitcher_mlbid", "season", "level_id", "game_type_group"]
NEW_COLS = ["rel_z", "rel_z_n", "rel_x", "rel_x_n", "ext", "ext_n", "arm_angle", "arm_angle_n"]

SEASONS = list(range(2015, 2027))


def backfill_season(season: int) -> None:
    pitcher_path = CHUNK_DIR / f"pitcher_stuff_new.season_{season}.parquet"
    pitch_types_path = CHUNK_DIR / f"new_pitch_types.season_{season}.parquet"

    if not pitcher_path.exists():
        print(f"  Skipping {season}: pitcher file not found")
        return
    if not pitch_types_path.exists():
        print(f"  Skipping {season}: pitch_types file not found")
        return

    pitchers = pl.read_parquet(pitcher_path).select(JOIN_KEYS + NEW_COLS)
    pitch_types = pl.read_parquet(pitch_types_path)

    already_present = [c for c in NEW_COLS if c in pitch_types.columns]
    if already_present:
        print(f"  {season}: dropping existing columns before rejoin: {already_present}")
        pitch_types = pitch_types.drop(already_present)

    updated = pitch_types.join(pitchers, on=JOIN_KEYS, how="left")
    updated.write_parquet(pitch_types_path)

    matched = updated.select(NEW_COLS[0]).drop_nulls().height
    print(f"  {season}: {matched}/{len(updated)} rows matched")


def recombine() -> None:
    chunks = sorted(CHUNK_DIR.glob("new_pitch_types.season_*.parquet"))
    if not chunks:
        print("No season chunks found to combine.")
        return
    combined = pl.concat([pl.scan_parquet(p) for p in chunks], how="diagonal_relaxed")
    out_path = OUT_DIR / "new_pitch_types.parquet"
    combined.sink_parquet(out_path)
    print(f"Wrote combined file: {out_path} ({len(chunks)} chunks)")


def main() -> None:
    print("Backfilling rel_z, rel_x, ext, arm_angle into new_pitch_types season chunks...")
    for season in SEASONS:
        print(f"Season {season}:")
        backfill_season(season)

    print("\nRecombining season chunks into new_pitch_types.parquet...")
    recombine()
    print("Done.")


if __name__ == "__main__":
    main()
