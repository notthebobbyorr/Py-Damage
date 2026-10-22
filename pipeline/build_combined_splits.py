"""Materialize observed split intersections, partitioned by season and level."""
from pathlib import Path
from itertools import combinations
import argparse
import sys
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data_aggregate import (
    _add_game_type_group, _normalize_player_names, _normalize_age_columns, _tag_pitch,
    _split_home_away, _split_month, _split_half, build_hitters, build_pitchers,
    build_pitch_types, compute_stuff_percentiles, apply_stuff_grade,
    build_team_pitching, _normalize_team_codes,
)
from pipeline.build_hitter_pitch_splits import SOURCE_COLUMNS, OPTIONAL_TRACKING
from pipeline.pitch_split_games import verified_games, apply_game_classifications
from app.hitter_pitch_splits import COLUMNS as HITTER_COLUMNS

from app.combined_split_config import DIMENSIONS, PITCH_COLUMNS
EXTRA_SOURCE = """pitcher_mlbid pitcher_hand pitcher_role pitcher_age pitcher_height name
pi_pitch_sub_type game_date home_team away_team half_inning home_mlbid away_mlbid throws stands
arm_angle arm_angle_right axis ext haa hbreak is_ball loc_adj_vaa outs_recorded pitching_code
pred_whiff_base primary_axis primary_hbreak primary_loc_adj_vaa primary_rpm primary_tag primary_vaa
primary_vbreak primary_velo primary_x_release primary_z_release release_x release_z rpm spin_efficiency
stuff_raw vaa vbreak x_angle_release z_angle_release pitch_group pitch_tag""".split()


def dimension_sets(hitter):
    dims = list(DIMENSIONS.values()) if hitter else list(DIMENSIONS.values())[:-1]
    return [list(c) for n in range(2, len(dims) + 1) for c in combinations(dims, n)
            if not {"split_half", "split_month"}.issubset(c)]


def add_dimensions(pitch, hitter):
    half = _split_half(pitch)
    pitch = half.rename({"__split": "split_half"})
    hand = "throws" if hitter else "stands"
    pitch = pitch.with_columns(
        pl.col(hand).alias("split_hand"),
        _split_home_away(pitch, hitter).alias("split_location"),
        _split_month(pitch).alias("split_month"),
        pl.when((pl.col("pi_pitch_group") == "FA") & pl.col("pi_pitch_type").is_in(["SI", "FT"]))
        .then(pl.lit("Sinker")).when((pl.col("pi_pitch_group") == "FA") & pl.col("pi_pitch_type").is_in(["FA", "FC"]))
        .then(pl.lit("Fastball"))
        .when(pl.col("pi_pitch_group").is_in(["SL", "CU"])).then(pl.lit("Breaking"))
        .when(pl.col("pi_pitch_group") == "CH").then(pl.lit("Offspeed"))
        .otherwise(pl.lit("Other")).alias("split_pitch_group"),
    )
    return pitch.with_columns(pl.when(pl.col("split_hand").is_in(["L", "R"]))
                              .then(pl.col("split_hand")).otherwise(None).alias("split_hand"))


def build_intersection(pitch, dims, kind, calibration):
    pitch = pitch.drop_nulls(dims)
    if pitch.is_empty():
        return pl.DataFrame()
    if kind.startswith("team_"):
        return build_team_intersection(pitch,dims,kind,calibration)
    hitter = kind == "hitter"
    keys = (["batter_mlbid", "hitter_name"] if hitter else ["pitcher_mlbid", "name", "pitcher_hand"])
    keys += ["season", "level_id", "game_type_group"] + dims
    if hitter:
        table = build_hitters(pitch, group_keys=keys)
        denominator_keys = [k for k in keys if k != "split_pitch_group"]
        totals = pitch.group_by(denominator_keys).agg(pl.len().alias("total_pitches"))
        table = table.join(totals, on=denominator_keys).with_columns(
            (100*pl.col("pitches")/pl.col("total_pitches")).alias("pitch_pct"),
            pl.when(pl.col("whiff_rate_den")>0).then(100*pl.col("whiff_rate_num")/pl.col("whiff_rate_den"))
            .otherwise(None).alias("whiff_pct"))
        wanted = keys + ["team", "PA", "max_EV"] + list(HITTER_COLUMNS)
    else:
        profiles = build_pitchers(pitch, extra_group_cols=dims)
        pitches = apply_stuff_grade(build_pitch_types(pitch, extra_group_cols=dims), calibration)
        if "stuff" not in pitches.columns:
            pitches = pitches.with_columns(pl.lit(None,dtype=pl.Float64).alias("stuff"))
        weighted = pitches.filter(pl.col("stuff").is_not_null()).group_by(keys).agg(
            ((pl.col("stuff")*pl.col("pitches")).sum()/pl.col("pitches").sum()).alias("stuff"))
        profiles = profiles.join(weighted, on=keys, how="left")
        if kind == "pitcher":
            table = profiles
        else:
            table = pitches.join(profiles.select(keys+["TBF", "GS", "IP"]), on=keys)
        wanted = keys + (["pitch_tag"] if kind == "pitch" else []) + ["team"] + list(PITCH_COLUMNS)
    return table.select(list(dict.fromkeys(c for c in wanted + ["miss_distance_n"] if c in table.columns)))


def build_team_intersection(pitch, dims, kind, calibration):
    hitting = kind == "team_hitter"
    team_col = "hitting_code" if hitting else "pitching_code"
    pitch = _normalize_team_codes(pitch,team_col).with_columns(pl.col(team_col).replace("OAK","ATH"))
    pitch = pitch.filter(pl.col(team_col).is_not_null() & ~pl.col(team_col).str.contains(r"^\d+$"))
    if pitch.is_empty():return pl.DataFrame()
    keys=[team_col,"season","level_id","game_type_group"]+dims
    if hitting:
        table=build_hitters(pitch,group_keys=keys)
        totals=pitch.group_by([k for k in keys if k!="split_pitch_group"]).agg(pl.len().alias("total_pitches"))
        table=table.join(totals,on=[k for k in keys if k!="split_pitch_group"]).with_columns(
            (100*pl.col("pitches")/pl.col("total_pitches")).alias("pitch_pct"),
            pl.when(pl.col("whiff_rate_den")>0).then(100*pl.col("whiff_rate_num")/pl.col("whiff_rate_den")).otherwise(None).alias("whiff_pct"))
        wanted=keys+["PA"]+list(HITTER_COLUMNS)
    else:
        table=build_team_pitching(pitch,extra_group_cols=dims)
        starts=pitch.filter(pl.col("pitcher_role")=="SP").group_by(keys).agg(pl.struct("game_pk","pitcher_mlbid").n_unique().alias("GS"))
        table=table.join(starts,on=keys,how="left").with_columns(pl.col("GS").fill_null(0))
        types=apply_stuff_grade(pitch.group_by(keys+["pitch_tag"]).agg(pl.mean("stuff_raw").alias("stuff_raw"),pl.len().alias("pitches")),calibration)
        if "stuff" in types:
            grades=types.filter(pl.col("stuff").is_not_null()).group_by(keys).agg(((pl.col("stuff")*pl.col("pitches")).sum()/pl.col("pitches").sum()).alias("stuff"))
            table=table.join(grades,on=keys,how="left")
        wanted=keys+list(PITCH_COLUMNS)
    return table.select(list(dict.fromkeys(c for c in wanted + ["miss_distance_n"] if c in table.columns)))


def write_partition(frame, path):
    if frame.is_empty():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.with_columns(pl.col(pl.Float64).cast(pl.Float32), pl.col(pl.String).cast(pl.Categorical))
    temp = path.with_suffix(".tmp.parquet")
    frame.write_parquet(temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--seasons", type=int, nargs="+")
    parser.add_argument("--kind", choices=["all", "hitter", "pitcher", "pitch", "team_hitter", "team_pitcher", "teams"], default="all")
    args = parser.parse_args()
    scans = [pl.scan_parquet(p) for p in args.input]
    verified = verified_games(scans)
    for raw in scans:
        scan = apply_game_classifications(raw, verified).filter(pl.col("game_type")=="R")
        schema = scan.collect_schema()
        cols = list(dict.fromkeys(c for c in SOURCE_COLUMNS+EXTRA_SOURCE if c in schema))
        for season in sorted(scan.select("season").unique().collect()["season"].to_list()):
            if args.seasons and season not in args.seasons:
                continue
            season_scan = scan.filter(pl.col("season")==season)
            calibration_cols = ["season", "level_id", "pitcher_mlbid", "pi_pitch_type", "pi_pitch_sub_type", "pi_pitch_group", "stuff_raw", "game_type"]
            calibration = compute_stuff_percentiles(_tag_pitch(_add_game_type_group(season_scan.select(calibration_cols).collect())))
            for level in season_scan.select("level_id").unique().collect()["level_id"].to_list():
                if level not in [1, 11, 14, 16]:
                    continue
                pitch = _normalize_age_columns(_normalize_player_names(_add_game_type_group(
                    season_scan.filter(pl.col("level_id")==level).select(cols).collect())))
                pitch = _tag_pitch(pitch).with_columns([
                    (pl.col(c).cast(pl.Float64,strict=False) if c in pitch.columns else pl.lit(None,dtype=pl.Float64)).alias(c)
                    for c in OPTIONAL_TRACKING+["arm_angle", "arm_angle_right", "pitcher_height"]])
                pitch = pitch.with_columns(pl.col("hitter_name").first().over(["batter_mlbid","season","level_id"]).alias("hitter_name"),
                                           pl.col("name").first().over(["pitcher_mlbid","season","level_id"]).alias("name"))
                kinds = ["hitter","pitcher","pitch","team_hitter","team_pitcher"] if args.kind=="all" else ["team_hitter","team_pitcher"] if args.kind=="teams" else [args.kind]
                for kind in kinds:
                    ready = add_dimensions(pitch, kind in {"hitter","team_hitter"})
                    for dims in dimension_sets(kind in {"hitter","team_hitter"}):
                        table = build_intersection(ready,dims,kind,calibration)
                        path = ROOT/"data/output/combined_splits"/kind/"__".join(dims)/f"season_{season}.level_{level}.parquet"
                        write_partition(table,path)
                    print(f"Built {kind} combined splits: {season}, level {level}",flush=True)
    if args.kind in {"all", "pitch"}:
        from pipeline.combined_execution import update_execution
        update_execution(args.input,args.seasons)


if __name__ == "__main__":
    main()
