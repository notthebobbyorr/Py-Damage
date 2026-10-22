"""Grade combined pitch slices from saved pitch-level execution predictions."""
from pathlib import Path
import sys
import shutil
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def grade_slices(scored, dims, calibration):
    keys=["pitcher_mlbid","pitcher_hand","season","level_id","game_type_group",*dims,"pitch_tag"]
    means=scored.drop_nulls(dims).group_by(keys).agg(pl.col("pred_v13").mean().alias("pred_v13"),pl.col("pred_v13").count().alias("execution_n"))
    pooled=calibration.group_by("pitch_tag").agg(pl.median("p1").alias("pool1"),pl.median("p99").alias("pool99"))
    means=means.join(calibration,on=["season","pitch_tag"],how="left").join(pooled,on="pitch_tag",how="left")
    means=means.with_columns(pl.coalesce("p1","pool1").alias("p1"),pl.coalesce("p99","pool99").alias("p99"))
    valid=(pl.col("execution_n")>=5)&(pl.col("p99")>pl.col("p1"))
    return means.with_columns(pl.when(valid).then((80-60*(pl.col("pred_v13")-pl.col("p1"))/(pl.col("p99")-pl.col("p1"))).clip(20,80).round(1)).otherwise(None).alias("grade_v13")).select(keys+["grade_v13","execution_n"])


def update_execution(inputs, seasons=None):
    from pipeline.build_combined_splits import add_dimensions, dimension_sets, write_partition
    from pipeline.pitch_split_games import verified_games, apply_game_classifications
    from pipeline.data_aggregate import _add_game_type_group
    scored_path=ROOT/"data/raw/location_v13_scored.parquet"
    if not scored_path.exists():
        print("No saved execution predictions; combined Execution Grades remain blank.")
        return
    scans=[pl.scan_parquet(p) for p in inputs]
    verified=verified_games(scans)
    scored=apply_game_classifications(pl.scan_parquet(scored_path),verified).filter(pl.col("game_type")=="R")
    calibration=pl.read_csv(ROOT/"models/location_v13_grade_calibration.csv")
    columns="game_pk pitcher_mlbid pitcher_hand season level_id game_type game_date home_team away_team half_inning home_mlbid away_mlbid hitting_code pitching_code throws stands pi_pitch_group pi_pitch_type".split()
    for year in scored.select("season").unique().collect()["season"].to_list():
        if seasons and year not in seasons:
            continue
        frames=[]
        for scan in scans:
            frame=apply_game_classifications(scan,verified).filter((pl.col("season")==year)&(pl.col("game_type")=="R")).select(columns).collect()
            if not frame.is_empty():frames.append(add_dimensions(frame,False))
        if not frames:continue
        raw=pl.concat(frames,how="diagonal_relaxed")
        join_keys=["game_pk","pitcher_mlbid","pitcher_hand","season","level_id"]
        metadata=raw.select(join_keys+["split_location","split_half","split_month"]).unique()
        if metadata.select(join_keys).is_duplicated().any():
            raise ValueError("Ambiguous game split metadata; execution grades were not updated.")
        predictions=scored.filter(pl.col("season")==year).collect().rename({"throws":"pitcher_hand","stands":"split_hand"})
        predictions=predictions.cast({k:metadata.schema[k] for k in join_keys})
        predictions=_add_game_type_group(predictions.join(metadata,on=join_keys,how="inner",validate="m:1"))
        for dims in dimension_sets(False):
            grades=grade_slices(predictions,dims,calibration)
            keys=[c for c in grades.columns if c not in ["grade_v13","execution_n"]]
            folder=ROOT/"data/output/combined_splits/pitch"/"__".join(dims)
            for path in folder.glob(f"season_{year}.level_*.parquet"):
                table=pl.read_parquet(path)
                subset=grades.filter(pl.col("level_id").is_in(table["level_id"].unique().cast(pl.Int64).implode())).cast({k:table.schema[k] for k in keys})
                table=table.drop([c for c in ["grade_v13","execution_n"] if c in table])
                result=table.join(subset,on=keys,how="left",validate="m:1")
                backup=ROOT/"data/backups/combined_execution"/path.relative_to(ROOT/"data/output")
                backup.parent.mkdir(parents=True,exist_ok=True)
                if not backup.exists():shutil.copy2(path,backup)
                write_partition(result,path)
        print(f"Execution grades updated: {year}",flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,nargs="+",required=True)
    args=parser.parse_args()
    update_execution(args.input)
