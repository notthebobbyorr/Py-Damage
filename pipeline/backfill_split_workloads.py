"""Add observed swing rates and workload counts without recalculating other metrics."""
from pathlib import Path
import argparse
import shutil
import sys
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.data_aggregate import _tag_pitch, _add_game_type_group, _split_vs_lr, _split_home_away, _split_month, _split_half, _with_split, _normalize_team_codes, add_percentiles
from pipeline.pitch_split_games import verified_games, apply_game_classifications

OUT = ROOT / "data/output"


def workload(pitch, keys, pitching):
    exprs = [pl.len().alias("pitches"), pl.sum("bbe").alias("bbe"),
             (100*(pl.col("swing")==1).sum()/pl.len()).alias("Swing_pct"),
             (pl.col("swing")==1).sum().alias("Swing_pct_num"), pl.len().alias("Swing_pct_den"),
             pl.col("pa_id").n_unique().alias("TBF" if pitching else "PA")]
    if pitching:
        exprs += [pl.col("game_pk").filter(pl.col("pitcher_role")=="SP").n_unique().alias("GS"),
                  (pl.col("outs_recorded").sum()/3).round(1).alias("IP")]
    return pitch.group_by(keys).agg(exprs)


def augment(frame, counts, keys):
    # Preserve established counts; replace only observed swing columns or missing fields.
    wanted = [c for c in counts.columns if c not in keys and (c.startswith("Swing_pct") or c not in frame.columns or frame[c].null_count())]
    if not wanted:
        return frame
    original = frame.height
    preserved = {c:frame[c] for c in wanted if c in frame.columns and not c.startswith("Swing_pct")}
    frame = frame.drop([c for c in wanted if c in frame.columns])
    # Source archives include levels outside the app's small integer key range.
    # Remove identities absent from this table before narrowing join key types.
    for key in keys:
        counts = counts.filter(pl.col(key).is_in(frame[key].unique().cast(counts.schema[key]).implode()))
    counts = counts.cast({k:frame.schema[k] for k in keys})
    frame = frame.join(counts.select(keys+wanted), on=keys, how="left", validate="m:1", maintain_order="left")
    frame = frame.with_columns([pl.coalesce(value,pl.col(column)).alias(column) for column,value in preserved.items()])
    if frame.height != original:
        raise ValueError("Workload join changed row count")
    return frame


def save(frame,path):
    backup = ROOT/"data/backups/split_workloads"/path.relative_to(OUT)
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():
        shutil.copy2(path,backup)
    temp = path.with_suffix(".tmp.parquet")
    frame.write_parquet(temp)
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,nargs="+",required=True)
    parser.add_argument("--seasons",type=int,nargs="+")
    args=parser.parse_args()
    scans=[pl.scan_parquet(p) for p in args.input]
    verified=verified_games(scans)
    cols="pitcher_mlbid batter_mlbid season level_id game_type pi_pitch_group pi_pitch_type pi_pitch_sub_type game_pk game_date home_team away_team half_inning home_mlbid away_mlbid throws stands hitting_code pitching_code swing bbe pa_id pitcher_role outs_recorded".split()
    collected={}
    group_pas={"hitter":[],"team":[],"league":[]}
    for scan in scans:
        schema=scan.collect_schema()
        for year in sorted(scan.select("season").unique().collect()["season"].to_list()):
            if args.seasons and year not in args.seasons:
                continue
            pitch=_tag_pitch(_add_game_type_group(scan.filter(pl.col("season")==year).select([c for c in cols if c in schema]).collect()))
            for kind in ["hitter","pitcher","pitch","team_hitter","team_pitcher"]:
                pitching=kind not in ["hitter","team_hitter"]
                identity={"hitter":"batter_mlbid","pitcher":"pitcher_mlbid","pitch":"pitcher_mlbid","team_hitter":"hitting_code","team_pitcher":"pitching_code"}[kind]
                ready=_normalize_team_codes(pitch,identity) if kind.startswith("team") else pitch
                keys=[identity,"season","level_id","game_type_group"]
                count_keys=keys+(["pitch_tag"] if kind=="pitch" else [])
                def counts_for(part):
                    result=workload(part,count_keys,pitching)
                    if kind=="pitch":
                        context=workload(part,keys,True).select(keys+["TBF","GS","IP"])
                        result=result.drop("TBF","GS","IP").join(context,on=keys)
                    return result
                collected.setdefault(kind,[]).append(counts_for(ready))
                parts=[]
                for split_type,split in [("vs L/R",_with_split(ready,_split_vs_lr(ready,"stands" if pitching else "throws"))),
                        ("Home/Away",_with_split(ready,_split_home_away(ready,not pitching))),
                        ("Monthly",_with_split(ready,_split_month(ready))), ("1st Half/2nd Half",_split_half(ready))]:
                    if split is None:
                        continue
                    for label in split["__split"].drop_nulls().unique():
                        parts.append(counts_for(split.filter(pl.col("__split")==label)).with_columns(pl.lit(split_type).alias("split_type"),pl.lit(label).alias("split")))
                collected.setdefault(kind+"_splits",[]).append(pl.concat(parts,how="diagonal_relaxed"))
            corrected=apply_game_classifications(pitch.lazy(),verified).filter(pl.col("game_type")=="R").collect()
            corrected=corrected.with_columns(pl.when((pl.col("pi_pitch_group")=="FA") & pl.col("pi_pitch_type").is_in(["SI","FT"])).then(pl.lit("SI"))
                .when((pl.col("pi_pitch_group")=="FA") & pl.col("pi_pitch_type").is_in(["FA","FC"])).then(pl.lit("FA"))
                .when(pl.col("pi_pitch_group").is_in(["SL","CU"])).then(pl.lit("BR"))
                .when(pl.col("pi_pitch_group")=="CH").then(pl.lit("OFF")).otherwise(None).alias("pitch_group"))
            for scope,ids in [("hitter",["batter_mlbid"]),("team",["hitting_code"]),("league",[])]:
                ready=corrected
                if scope=="team":
                    ready=_normalize_team_codes(ready,"hitting_code").with_columns(pl.col("hitting_code").replace("OAK","ATH"))
                group_pas[scope].append(ready.group_by(ids+["season","level_id","pitch_group"]).agg(pl.col("pa_id").n_unique().alias("PA")))
            print(f"Counted workloads: {year}",flush=True)
    mapping={"hitter":["damage_pos_*"],"pitcher":["pitcher_stuff_new","pitcher_pctiles","pitchers_regressed"],
        "pitch":["new_pitch_types","pitch_types_pctiles","pitch_types_regressed"],"team_hitter":["new_team_damage"],"team_pitcher":["new_team_stuff"],
        "hitter_splits":["hitter_splits"],"pitcher_splits":["pitcher_splits"],"pitch_splits":["pitch_types_splits"],
        "team_hitter_splits":["team_hitter_splits"],"team_pitcher_splits":["team_pitcher_splits"]}
    for kind,patterns in mapping.items():
        counts=pl.concat(collected[kind],how="diagonal_relaxed")
        identity=next(c for c in counts.columns if c in {"pitcher_mlbid","batter_mlbid","hitting_code","pitching_code"})
        keys=[identity,"season","level_id","game_type_group"]+(["pitch_tag"] if kind in {"pitch","pitch_splits"} else [])+(["split_type","split"] if kind.endswith("_splits") else [])
        paths=set()
        for pattern in patterns:
            paths.update(OUT.glob(pattern+".parquet"))
            paths.update((OUT/"_season_chunks").glob(pattern+".season_*.parquet"))
        for path in sorted(paths):
            frame=pl.read_parquet(path)
            if not set(keys).issubset(frame.columns):
                continue
            # On daily runs retain already-populated historical rows verbatim.
            years=counts["season"].unique()
            selected=frame.filter(pl.col("season").is_in(years.implode()))
            rest=frame.filter(~pl.col("season").is_in(years.implode()))
            selected=augment(selected,counts,keys)
            if path.name.startswith("pitch_types_pctiles"):
                selected=add_percentiles(selected.drop([c for c in ["Swing_pct_pctile"] if c in selected]),["season","level_id","game_type_group","pitch_tag"],["Swing_pct"],"pitches",100)
            save(pl.concat([rest,selected],how="diagonal_relaxed"),path)
    for scope,parts in group_pas.items():
        counts=pl.concat(parts,how="diagonal_relaxed")
        path=OUT/("hitter_pitch_splits.parquet" if scope=="hitter" else f"{scope}_hitting_pitch_splits.parquet")
        if path.exists():
            keys=(["batter_mlbid"] if scope=="hitter" else ["hitting_code"] if scope=="team" else [])+["season","level_id","pitch_group"]
            save(augment(pl.read_parquet(path),counts,keys),path)
    print("Saved observed swing rates and workload counts.",flush=True)


if __name__=="__main__":
    main()
